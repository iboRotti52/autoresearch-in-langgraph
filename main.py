import argparse
import ast
import difflib
import json
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from prepare import EVAL_TOKENS, TIME_BUDGET
from typing_extensions import Literal, NotRequired, TypedDict


PROJECT_DIR = Path(__file__).resolve().parent
MEMORY_DIR = PROJECT_DIR / "memory"
RESEARCH_FILE = MEMORY_DIR / "research.jsonl"
EXPERIMENT_FILE = MEMORY_DIR / "experiments.jsonl"
CODE_SNAPSHOT_DIR = MEMORY_DIR / "code_snapshots"
RUNS_DIR = MEMORY_DIR / "runs"
RESEARCH_SCHEMA_FILE = PROJECT_DIR / "research_schema.json"
PLANNER_SCHEMA_FILE = PROJECT_DIR / "planner_schema.json"
CODE_MODIFICATION_SCHEMA_FILE = PROJECT_DIR / "code_modification_schema.json"
EXPERIMENT_TIMEOUT_SECONDS = TIME_BUDGET + 3600
CODEX_RESEARCH_TIMEOUT_SECONDS = 900
CLAUDE_TIMEOUT_SECONDS = 300
MODEL_PROVIDER: Literal["claude", "codex"] = "claude"

DEFAULT_CONSTRAINTS = [
    "prepare.py cannot be modified.",
    "The dataset cannot be modified.",
    "The evaluation metric cannot be modified.",
    f"The training budget is fixed at {TIME_BUDGET} seconds.",
    f"Evaluation uses exactly {EVAL_TOKENS} tokens.",
    "Experiments must remain comparable.",
    "Do not introduce new dependencies without a deliberate design change.",
    "Every experiment must use the same experiment budget.",
    "Research direction must come from Researcher, not Modify Code.",
    "Implementation planning must come from Planner, not Modify Code.",
]


class ResearchIdea(TypedDict):
    id: str
    idea: str
    source: str
    why_selected: str
    evidence: str
    hypothesis: str
    expected_effect: str
    status: Literal["selected", "pending", "tested", "rejected"]


class ResearchProposal(TypedDict):
    idea: str
    source: str
    why_selected: str
    evidence: str
    hypothesis: str
    expected_effect: str


class ExperimentPlan(TypedDict):
    idea_id: str
    goal: str
    changes: list[str]
    scope: Literal["small", "structural"]
    reasoning: str


class ExperimentPlanProposal(TypedDict):
    goal: str
    changes: list[str]
    scope: Literal["small", "structural"]
    reasoning: str


class CodeEdit(TypedDict):
    start_line: int
    end_line: int
    new_code: str


class CodeModification(TypedDict):
    edits: list[CodeEdit]
    code_changes: list[str]
    code_change_reasoning: str


class AutoResearchState(TypedDict):
    editable_files: list[str]
    read_only_files: list[str]
    constraints: list[str]

    research_history: list[ResearchIdea]

    current_idea: NotRequired[ResearchIdea]
    experiment_plan: NotRequired[ExperimentPlan]

    original_code: NotRequired[str]
    code_snapshot: NotRequired[str]
    code_changes: NotRequired[list[str]]
    code_change_reasoning: NotRequired[str]
    added_lines: NotRequired[int]
    removed_lines: NotRequired[int]

    experiment_log_path: NotRequired[str]
    experiment_logs: NotRequired[list[str]]
    experiment_summary: NotRequired[str]
    experiment_error_tail: NotRequired[str]
    experiment_returncode: NotRequired[int | None]
    experiment_timed_out: NotRequired[bool]
    repair_attempts: NotRequired[int]
    repair_failed: NotRequired[bool]

    best_score: NotRequired[float]
    previous_score: NotRequired[float | None]
    current_score: NotRequired[float | None]
    peak_memory_mb: NotRequired[float | None]
    decision: NotRequired[Literal["keep", "revert"]]
    completed_experiments: NotRequired[int]
    max_experiments: NotRequired[int]


StateUpdate = dict[str, object]


def load_jsonl(file_path: Path) -> list[dict]:
    if not file_path.exists():
        return []

    with file_path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_research(idea: ResearchIdea) -> None:
    MEMORY_DIR.mkdir(exist_ok=True)

    with RESEARCH_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(idea, ensure_ascii=False) + "\n")


def save_experiment(experiment: dict) -> None:
    MEMORY_DIR.mkdir(exist_ok=True)

    with EXPERIMENT_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(experiment, ensure_ascii=False) + "\n")


def load_last_kept_score() -> float | None:
    best_score = None

    for experiment in load_jsonl(EXPERIMENT_FILE):
        if (
            experiment.get("time_budget_seconds", 300) == TIME_BUDGET
            and experiment.get("eval_tokens", 40 * 524288) == EVAL_TOKENS
            and experiment.get("decision") == "keep"
            and experiment.get("execution_status") == "completed"
            and experiment.get("new_score") is not None
        ):
            best_score = float(experiment["new_score"])

    return best_score


def mark_research_as_tested(idea_id: str) -> None:
    research_history = load_jsonl(RESEARCH_FILE)

    for idea in research_history:
        if idea["id"] == idea_id:
            idea["status"] = "tested"

    with RESEARCH_FILE.open("w", encoding="utf-8") as file:
        for idea in research_history:
            file.write(json.dumps(idea, ensure_ascii=False) + "\n")


def configure_provider(provider: Literal["claude", "codex"]) -> None:
    global MODEL_PROVIDER

    if provider not in ("claude", "codex"):
        raise ValueError("Provider must be 'claude' or 'codex'.")

    MODEL_PROVIDER = provider


def run_claude(
    prompt: str,
    schema_file: Path,
    research_tools: bool = False,
) -> dict:
    """Run Claude CLI with the user's subscription and return validated JSON."""
    try:
        process = subprocess.run(
            [
                "claude",
                "--print",
                "--safe-mode",
                "--output-format",
                "json",
                "--json-schema",
                schema_file.read_text(encoding="utf-8"),
                "--tools",
                "Read,WebSearch,WebFetch" if research_tools else "",
                "--no-session-persistence",
            ],
            cwd=PROJECT_DIR,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=(
                CODEX_RESEARCH_TIMEOUT_SECONDS
                if research_tools
                else CLAUDE_TIMEOUT_SECONDS
            ),
        )
    except FileNotFoundError as error:
        raise RuntimeError("Claude CLI is not installed.") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Claude CLI timed out.") from error

    if process.returncode != 0:
        message = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"Claude CLI failed: {message}")

    try:
        response = json.loads(process.stdout)
        return response["structured_output"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(
            "Claude CLI did not return the expected structured output."
        ) from error


def run_codex(
    prompt: str,
    schema_file: Path,
    research_tools: bool = False,
) -> dict:
    with tempfile.TemporaryDirectory() as temp_directory:
        output_file = Path(temp_directory) / "result.json"
        command = ["codex"]

        if research_tools:
            command.append("--search")

        command.extend(
            [
                "exec",
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--skip-git-repo-check",
                "--output-schema",
                str(schema_file),
                "--output-last-message",
                str(output_file),
                "--cd",
                str(PROJECT_DIR),
                "-",
            ]
        )

        try:
            process = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=(
                    CODEX_RESEARCH_TIMEOUT_SECONDS
                    if research_tools
                    else CLAUDE_TIMEOUT_SECONDS
                ),
            )
        except FileNotFoundError as error:
            raise RuntimeError("Codex CLI is not installed.") from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Codex CLI timed out.") from error

        if process.returncode != 0:
            message = process.stderr.strip() or process.stdout.strip()
            raise RuntimeError(f"Codex CLI failed: {message}")

        try:
            return json.loads(output_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError) as error:
            raise RuntimeError(
                "Codex CLI did not return the expected structured output."
            ) from error


def run_model(
    prompt: str,
    schema_file: Path,
    research_tools: bool = False,
) -> dict:
    if MODEL_PROVIDER == "claude":
        return run_claude(prompt, schema_file, research_tools)

    return run_codex(prompt, schema_file, research_tools)


def researcher(state: AutoResearchState) -> StateUpdate:
    research_history = load_jsonl(RESEARCH_FILE)
    experiment_history = load_jsonl(EXPERIMENT_FILE)
    research_history_json = json.dumps(
        research_history,
        indent=2,
        ensure_ascii=False,
    )
    experiment_history_json = json.dumps(
        experiment_history,
        indent=2,
        ensure_ascii=False,
    )

    prompt = f"""
You are the Researcher in an autonomous ML research system.

Your job is to decide WHAT experiments are worth trying.
Do not write or modify code.
Inspect the listed project files and use live web search for relevant papers or
technical sources. Every external claim must include its real source URL.

EDITABLE FILES:
{state["editable_files"]}

READ ONLY FILES:
{state["read_only_files"]}

CONSTRAINTS:
{state["constraints"]}

PREVIOUS RESEARCH:
{research_history_json}

PREVIOUS EXPERIMENTS:
{experiment_history_json}

CURRENT PROTOCOL:
- training budget: {TIME_BUDGET} seconds
- evaluation tokens: {EVAL_TOKENS}
- history rows without protocol fields belong to the legacy 300-second run
- use legacy results as lessons, but do not compare their scores directly with
  the current protocol

Choose exactly one next experiment idea.

Learn from both successful and failed experiments. Prefer a useful follow-up to
a promising result or near-miss when one exists; otherwise choose a new direction.
Do not repeat a tested experiment.
Do not invent papers, URLs, or research results.
Prefer the smallest informative experiment. Consider simplification and memory
cost as well as val_bpb. Web research is optional, not mandatory.

For the one selected idea explain:
- idea: what the idea is
- source: the exact file, previous experiment, or provided paper and URL it came from
- why_selected: why you selected it
- evidence: the relevant evidence from that source
- hypothesis: a testable claim
- expected_effect: the expected effect on val_bpb
"""

    candidate: ResearchProposal = run_model(
        prompt,
        RESEARCH_SCHEMA_FILE,
        research_tools=True,
    )

    idea: ResearchIdea = {
        **candidate,
        "id": str(uuid.uuid4()),
        "status": "selected",
    }
    save_research(idea)

    return {
        "research_history": research_history + [idea],
        "current_idea": idea,
    }


def experiment_planner(state: AutoResearchState) -> StateUpdate:
    if "current_idea" not in state:
        raise RuntimeError("Planner needs a current idea.")

    current_idea = state["current_idea"]

    editable_code = {
        file_path: (PROJECT_DIR / file_path).read_text(encoding="utf-8")
        for file_path in state["editable_files"]
    }

    prompt = f"""
You are the Experiment Planner in an autonomous ML research system.

Convert the selected research idea into one concrete experiment plan.
The selected idea and its hypothesis are the only research direction.
Do not replace them with another idea. Do not write code.
Plan only the smallest changes needed to test that exact hypothesis.
Do not propose features that are already present in the current code.
Do not add new metrics, logging, or unrelated comparisons.
Respect all constraints.
Use scope "small" for normal experiments. Use "structural" only when the
hypothesis genuinely requires a larger architecture or training-loop change.

SELECTED IDEA:
{current_idea}

CONSTRAINTS:
{state["constraints"]}

CURRENT PROTOCOL:
- training budget: {TIME_BUDGET} seconds
- evaluation tokens: {EVAL_TOKENS}

EDITABLE CODE:
{editable_code}

Before returning, verify that every proposed change directly tests this exact idea:
{current_idea["idea"]}

Return:
- goal: what the experiment tests
- changes: concrete code changes required
- scope: small or structural
- reasoning: why these changes test the hypothesis
"""

    result: ExperimentPlanProposal = run_model(prompt, PLANNER_SCHEMA_FILE)

    experiment_plan: ExperimentPlan = {
        "idea_id": current_idea["id"],
        **result,
    }

    return {
        "current_idea": current_idea,
        "experiment_plan": experiment_plan,
    }


def apply_code_edits(
    original_code: str,
    result: CodeModification,
    file_path: Path,
    max_edits: int,
    max_replaced_lines: int,
    max_new_lines: int,
) -> str:
    original_lines = original_code.splitlines(keepends=True)
    edits = result["edits"]

    if not 1 <= len(edits) <= max_edits:
        raise RuntimeError(f"Modify Code may return at most {max_edits} edits.")

    previous_start = len(original_lines) + 1
    modified_lines = original_lines[:]

    for edit in sorted(edits, key=lambda item: item["start_line"], reverse=True):
        start_line = edit["start_line"]
        end_line = edit["end_line"]
        new_code = edit["new_code"]

        if not (1 <= start_line <= end_line <= len(original_lines)):
            raise RuntimeError("Modify Code returned an invalid line range.")
        if end_line - start_line + 1 > max_replaced_lines:
            raise RuntimeError("A code edit replaced too many existing lines.")
        if len(new_code.splitlines()) > max_new_lines:
            raise RuntimeError("A code edit introduced too many new lines.")
        if end_line >= previous_start:
            raise RuntimeError("Modify Code returned overlapping edits.")

        if original_lines[end_line - 1].endswith("\n") and not new_code.endswith("\n"):
            new_code += "\n"

        modified_lines[start_line - 1 : end_line] = [new_code]
        previous_start = start_line

    modified_code = "".join(modified_lines)
    ast.parse(modified_code, filename=str(file_path))
    return modified_code


def count_changed_lines(original_code: str, modified_code: str) -> tuple[int, int]:
    diff = difflib.ndiff(original_code.splitlines(), modified_code.splitlines())
    added_lines = sum(1 for line in diff if line.startswith("+ "))

    diff = difflib.ndiff(original_code.splitlines(), modified_code.splitlines())
    removed_lines = sum(1 for line in diff if line.startswith("- "))
    return added_lines, removed_lines


def modify_code(state: AutoResearchState) -> StateUpdate:
    if "current_idea" not in state or "experiment_plan" not in state:
        raise RuntimeError("Modify Code needs a current idea and experiment plan.")

    if len(state["editable_files"]) != 1:
        raise RuntimeError("Modify Code currently supports one editable file.")

    file_path = PROJECT_DIR / state["editable_files"][0]
    original_code = file_path.read_text(encoding="utf-8")
    scope = state["experiment_plan"]["scope"]

    if scope == "small":
        max_edits, max_replaced_lines, max_new_lines = 3, 20, 30
    else:
        max_edits, max_replaced_lines, max_new_lines = 5, 80, 80

    numbered_code = "\n".join(
        f"{line_number}: {line}"
        for line_number, line in enumerate(original_code.splitlines(), start=1)
    )

    prompt = f"""
You are the coding step in an autonomous ML research system.

Implement the experiment plan in the editable code.
Do not invent a new research direction.
Do not make unrelated improvements.
Respect all constraints.
Return non-overlapping edits within the limits below. Each edit contains the 1-based start
and end line numbers to replace plus replacement code without line-number
prefixes or Markdown fences.
Make the smallest possible change that implements the plan.

EDIT LIMITS:
- maximum edits: {max_edits}
- maximum existing lines replaced by one edit: {max_replaced_lines}
- maximum new lines introduced by one edit: {max_new_lines}

RESEARCH IDEA:
{state["current_idea"]}

EXPERIMENT PLAN:
{state["experiment_plan"]}

CONSTRAINTS:
{state["constraints"]}

EDITABLE FILE:
{file_path.name}

CURRENT CODE:
{numbered_code}
"""

    result: CodeModification = run_model(
        prompt,
        CODE_MODIFICATION_SCHEMA_FILE,
    )

    modified_code = apply_code_edits(
        original_code,
        result,
        file_path,
        max_edits,
        max_replaced_lines,
        max_new_lines,
    )
    added_lines, removed_lines = count_changed_lines(original_code, modified_code)

    CODE_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_file = (
        CODE_SNAPSHOT_DIR / f'{state["current_idea"]["id"]}_{file_path.name}'
    )
    snapshot_file.write_text(original_code, encoding="utf-8")
    file_path.write_text(modified_code, encoding="utf-8")

    return {
        "original_code": original_code,
        "code_snapshot": str(snapshot_file.relative_to(PROJECT_DIR)),
        "code_changes": result["code_changes"],
        "code_change_reasoning": result["code_change_reasoning"],
        "added_lines": added_lines,
        "removed_lines": removed_lines,
        "repair_attempts": 0,
        "repair_failed": False,
        "experiment_logs": [],
    }


def read_log_tail(log_path: Path, line_count: int) -> str:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def run_experiment(state: AutoResearchState) -> StateUpdate:
    train_file = state["editable_files"][0]
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if "current_idea" in state:
        run_name = state["current_idea"]["id"]
    else:
        run_name = f"baseline_{uuid.uuid4().hex[:8]}"

    attempt = state.get("repair_attempts", 0)
    log_file = RUNS_DIR / f"{run_name}_attempt{attempt}.log"
    relative_log_path = str(log_file.relative_to(PROJECT_DIR))

    try:
        with log_file.open("w", encoding="utf-8") as output_file:
            result = subprocess.run(
                [sys.executable, train_file],
                cwd=PROJECT_DIR,
                stdout=output_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=EXPERIMENT_TIMEOUT_SECONDS,
            )

        summary = read_log_tail(log_file, 12)
        error_tail = read_log_tail(log_file, 50) if result.returncode != 0 else ""

        return {
            "experiment_log_path": relative_log_path,
            "experiment_logs": state.get("experiment_logs", []) + [relative_log_path],
            "experiment_summary": summary,
            "experiment_error_tail": error_tail,
            "experiment_returncode": result.returncode,
            "experiment_timed_out": False,
            "current_score": None,
            "peak_memory_mb": None,
        }

    except subprocess.TimeoutExpired:
        return {
            "experiment_log_path": relative_log_path,
            "experiment_logs": state.get("experiment_logs", []) + [relative_log_path],
            "experiment_summary": read_log_tail(log_file, 12),
            "experiment_error_tail": read_log_tail(log_file, 50),
            "experiment_returncode": None,
            "experiment_timed_out": True,
            "current_score": None,
            "peak_memory_mb": None,
        }


def read_metric(state: AutoResearchState) -> StateUpdate:
    if "experiment_log_path" not in state:
        raise RuntimeError("Read Metric needs an experiment log.")

    log_file = PROJECT_DIR / state["experiment_log_path"]
    output = log_file.read_text(encoding="utf-8", errors="replace")
    score_matches = re.findall(
        r"val_bpb\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        output,
    )
    memory_matches = re.findall(
        r"peak_(?:vram|memory)_mb\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        output,
    )

    score = float(score_matches[-1]) if score_matches else None
    peak_memory = float(memory_matches[-1]) if memory_matches else None

    return {
        "current_score": score,
        "peak_memory_mb": peak_memory,
    }


def repair_code(state: AutoResearchState) -> StateUpdate:
    if "current_idea" not in state or "experiment_plan" not in state:
        raise RuntimeError("Repair Code needs the current experiment.")

    file_path = PROJECT_DIR / state["editable_files"][0]
    candidate_code = file_path.read_text(encoding="utf-8")
    numbered_code = "\n".join(
        f"{line_number}: {line}"
        for line_number, line in enumerate(candidate_code.splitlines(), start=1)
    )

    prompt = f"""
You are repairing one failed autonomous ML experiment.

Fix only a simple implementation error such as a typo, missing import, invalid
name, or obvious shape/API mismatch. Do not change the research idea, experiment
plan, or intended behavior. Do not add an unrelated improvement.

Return at most three non-overlapping edits. Each edit may replace at most 20
existing lines and introduce at most 30 new lines. If the idea cannot be repaired
without changing the hypothesis, do not attempt a redesign.

RESEARCH IDEA:
{state["current_idea"]}

EXPERIMENT PLAN:
{state["experiment_plan"]}

ERROR TAIL:
{state.get("experiment_error_tail", "")}

CURRENT CANDIDATE CODE:
{numbered_code}
"""

    try:
        result: CodeModification = run_model(
            prompt,
            CODE_MODIFICATION_SCHEMA_FILE,
        )
        repaired_code = apply_code_edits(
            candidate_code,
            result,
            file_path,
            max_edits=3,
            max_replaced_lines=20,
            max_new_lines=30,
        )
    except RuntimeError as error:
        return {
            "repair_attempts": 1,
            "repair_failed": True,
            "experiment_error_tail": (
                state.get("experiment_error_tail", "")
                + f"\nRepair failed: {error}"
            ).strip(),
        }

    file_path.write_text(repaired_code, encoding="utf-8")
    added_lines, removed_lines = count_changed_lines(
        state["original_code"],
        repaired_code,
    )

    return {
        "repair_attempts": 1,
        "repair_failed": False,
        "code_changes": state.get("code_changes", []) + result["code_changes"],
        "code_change_reasoning": (
            state.get("code_change_reasoning", "")
            + " Repair: "
            + result["code_change_reasoning"]
        ).strip(),
        "added_lines": added_lines,
        "removed_lines": removed_lines,
    }


def evaluate(state: AutoResearchState) -> StateUpdate:
    if state.get("current_score") is None:
        raise RuntimeError("Evaluate needs a current score.")

    best_score = state.get("best_score")
    current_score = state["current_score"]

    # Memory and changed-line counts are recorded now. They will become decision
    # criteria only after a real baseline establishes useful thresholds.

    if best_score is None or current_score < best_score:
        return {
            "previous_score": best_score,
            "decision": "keep",
        }

    return {
        "previous_score": best_score,
        "decision": "revert",
    }


def keep(state: AutoResearchState) -> StateUpdate:
    if state.get("decision") != "keep":
        raise RuntimeError("Keep node requires a keep decision.")

    if state.get("current_score") is None:
        raise RuntimeError("Keep node requires a current score.")

    return {"best_score": state["current_score"]}


def revert(state: AutoResearchState) -> StateUpdate:
    if state.get("decision") != "revert":
        raise RuntimeError("Revert node requires a revert decision.")

    if "code_snapshot" not in state:
        raise RuntimeError("Revert node requires a code snapshot.")

    if len(state["editable_files"]) != 1:
        raise RuntimeError("Revert currently supports one editable file.")

    file_path = PROJECT_DIR / state["editable_files"][0]
    snapshot_file = PROJECT_DIR / state["code_snapshot"]
    file_path.write_text(snapshot_file.read_text(encoding="utf-8"), encoding="utf-8")

    return {}


def record_result(state: AutoResearchState) -> StateUpdate:
    required_fields = [
        "current_idea",
        "experiment_plan",
        "code_changes",
        "code_change_reasoning",
        "decision",
    ]
    if any(field not in state for field in required_fields):
        raise RuntimeError("Record Result is missing experiment information.")

    if state.get("experiment_timed_out"):
        execution_status = "timeout"
    elif state.get("experiment_returncode") != 0:
        execution_status = "failed"
    elif state.get("current_score") is None:
        execution_status = "metric_missing"
    else:
        execution_status = "completed"

    current_idea = state["current_idea"]
    experiment = {
        "type": "experiment",
        "time_budget_seconds": TIME_BUDGET,
        "eval_tokens": EVAL_TOKENS,
        "idea_id": current_idea["id"],
        "hypothesis": current_idea["hypothesis"],
        "experiment_plan": state["experiment_plan"],
        "code_changes": state["code_changes"],
        "code_change_reasoning": state["code_change_reasoning"],
        "code_snapshot": state.get("code_snapshot"),
        "added_lines": state.get("added_lines", 0),
        "removed_lines": state.get("removed_lines", 0),
        "repair_attempts": state.get("repair_attempts", 0),
        "previous_score": state.get("previous_score"),
        "new_score": state.get("current_score"),
        "peak_memory_mb": state.get("peak_memory_mb"),
        "decision": state["decision"],
        "execution_status": execution_status,
        "run_logs": state.get("experiment_logs", []),
    }

    save_experiment(experiment)
    mark_research_as_tested(current_idea["id"])

    tested_idea: ResearchIdea = {**current_idea, "status": "tested"}
    research_history = [
        tested_idea if idea["id"] == tested_idea["id"] else idea
        for idea in state["research_history"]
    ]

    return {
        "current_idea": tested_idea,
        "research_history": research_history,
        "completed_experiments": state.get("completed_experiments", 0) + 1,
    }


def route_after_evaluate(
    state: AutoResearchState,
) -> Literal["keep", "revert"]:
    return state["decision"]


def route_after_run(
    state: AutoResearchState,
) -> Literal["read_metric", "repair_code", "experiment_failed"]:
    if state["experiment_timed_out"]:
        return "experiment_failed"

    if state["experiment_returncode"] != 0:
        error_tail = state.get("experiment_error_tail", "").lower()
        non_repairable = ("out of memory", "oom", "killed")

        if (
            state.get("repair_attempts", 0) < 1
            and not any(pattern in error_tail for pattern in non_repairable)
        ):
            return "repair_code"

        return "experiment_failed"

    return "read_metric"


def route_after_repair(
    state: AutoResearchState,
) -> Literal["run_experiment", "experiment_failed"]:
    if state.get("repair_failed"):
        return "experiment_failed"

    return "run_experiment"


def route_after_metric(
    state: AutoResearchState,
) -> Literal["evaluate", "experiment_failed"]:
    if state.get("current_score") is None:
        return "experiment_failed"

    return "evaluate"


def mark_experiment_failed(state: AutoResearchState) -> StateUpdate:
    return {
        "previous_score": state.get("best_score"),
        "current_score": None,
        "decision": "revert",
    }


def route_after_record(
    state: AutoResearchState,
) -> Literal["researcher", "end"]:
    max_experiments = state.get("max_experiments")
    if (
        max_experiments is not None
        and state.get("completed_experiments", 0) >= max_experiments
    ):
        return "end"

    return "researcher"


def build_graph():
    workflow = StateGraph(AutoResearchState)

    workflow.add_node("researcher", researcher)
    workflow.add_node("experiment_planner", experiment_planner)
    workflow.add_node("modify_code", modify_code)
    workflow.add_node("run_experiment", run_experiment)
    workflow.add_node("read_metric", read_metric)
    workflow.add_node("repair_code", repair_code)
    workflow.add_node("evaluate", evaluate)
    workflow.add_node("experiment_failed", mark_experiment_failed)
    workflow.add_node("keep", keep)
    workflow.add_node("revert", revert)
    workflow.add_node("record_result", record_result)

    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "experiment_planner")
    workflow.add_edge("experiment_planner", "modify_code")
    workflow.add_edge("modify_code", "run_experiment")

    workflow.add_conditional_edges(
        "run_experiment",
        route_after_run,
        {
            "read_metric": "read_metric",
            "repair_code": "repair_code",
            "experiment_failed": "experiment_failed",
        },
    )

    workflow.add_conditional_edges(
        "repair_code",
        route_after_repair,
        {
            "run_experiment": "run_experiment",
            "experiment_failed": "experiment_failed",
        },
    )

    workflow.add_conditional_edges(
        "read_metric",
        route_after_metric,
        {
            "evaluate": "evaluate",
            "experiment_failed": "experiment_failed",
        },
    )

    workflow.add_edge("experiment_failed", "revert")

    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "keep": "keep",
            "revert": "revert",
        },
    )

    workflow.add_edge("keep", "record_result")
    workflow.add_edge("revert", "record_result")

    workflow.add_conditional_edges(
        "record_result",
        route_after_record,
        {
            "researcher": "researcher",
            "end": END,
        },
    )

    return workflow.compile()


autoresearch_graph = build_graph()


def create_initial_state(
    max_experiments: int | None = None,
) -> AutoResearchState:
    research_history = load_jsonl(RESEARCH_FILE)
    best_score = load_last_kept_score()

    state: AutoResearchState = {
        "editable_files": ["train.py"],
        "read_only_files": ["prepare.py"],
        "constraints": DEFAULT_CONSTRAINTS,
        "research_history": research_history,
        "completed_experiments": 0,
    }

    if best_score is not None:
        state["best_score"] = best_score
    if max_experiments is not None:
        state["max_experiments"] = max_experiments

    return state


def run_baseline() -> float:
    if load_last_kept_score() is not None:
        raise RuntimeError("A kept baseline or experiment already exists.")

    result = run_experiment(create_initial_state())

    if result["experiment_timed_out"] or result["experiment_returncode"] != 0:
        raise RuntimeError(
            "Baseline failed:\n" + str(result["experiment_error_tail"])
        )

    metric = read_metric(result)
    if metric["current_score"] is None:
        raise RuntimeError("Baseline completed without a val_bpb metric.")

    save_experiment(
        {
            "type": "baseline",
            "time_budget_seconds": TIME_BUDGET,
            "eval_tokens": EVAL_TOKENS,
            "previous_score": None,
            "new_score": metric["current_score"],
            "peak_memory_mb": metric["peak_memory_mb"],
            "decision": "keep",
            "execution_status": "completed",
            "run_logs": result["experiment_logs"],
        }
    )

    return metric["current_score"]


def run_graph(max_experiments: int) -> None:
    if max_experiments < 1:
        raise RuntimeError("max_experiments must be at least 1.")

    initial_state = create_initial_state(max_experiments)
    if initial_state.get("best_score") is None:
        raise RuntimeError("Run the baseline command before starting experiments.")

    try:
        for event in autoresearch_graph.stream(
            initial_state,
            config={"recursion_limit": max_experiments * 15 + 10},
        ):
            print(json.dumps(event, indent=2, ensure_ascii=False))
    except GraphRecursionError as error:
        raise RuntimeError("Graph did not finish an experiment safely.") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("baseline")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--max-experiments", type=int, default=1)

    arguments = parser.parse_args()

    try:
        if arguments.command == "baseline":
            print(f"Baseline val_bpb: {run_baseline():.6f}")
        else:
            run_graph(arguments.max_experiments)
    except RuntimeError as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
