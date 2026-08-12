# AutoResearch in LangGraph — DESIGN.md

## 1. Goal

Reimplement the core behavior of Karpathy's AutoResearch using LangGraph as a learning project.

The goal is:

> Karpathy AutoResearch behavior + explicit LangGraph orchestration + minimal useful persistent memory.

This project is separate from the future Turkish AutoResearch project.

The system should remain relatively minimal and should not become an unnecessarily complex multi-agent research platform.

---

## 2. Core Philosophy

Karpathy's original loop is approximately:

```text
Read instructions
→ inspect train.py
→ think of an experiment
→ modify train.py
→ run training
→ read val_bpb
→ keep or revert
→ repeat
```

Our version makes some of these implicit responsibilities explicit through LangGraph nodes.

---

## 3. Graph

Current intended graph:

```text
START
  ↓
Researcher
  ↓
Experiment Planner
  ↓
Modify Code
  ↓
Run Experiment
  ├── simple runtime crash → Repair Code → Run Experiment (once)
  ├── timeout/OOM/second crash → Revert
  ↓ success
Read Metric
  ↓
Evaluate
 ↙       ↘
KEEP    REVERT
  ↘       ↙
Record Result
     ↓
Researcher
     ↓
choose exactly one next idea using all previous results
```

Important routing rule:

**Return to Researcher after every completed experiment.**

There is no pending idea queue. Researcher sees the current accepted `train.py`,
all persisted ideas, and all experiment outcomes, then selects exactly one next
idea. This lets each decision learn from the immediately preceding result.

---

# 4. Node Responsibilities

## Researcher

Purpose:

> Decide **what is worth trying**.

The Researcher does not modify code.

Inputs should include:

- editable file information
- read-only file information
- experiment constraints
- relevant source code
- previous research
- previous experiments

The Researcher selects exactly one useful next idea in each research pass.

The Researcher should avoid proposing ideas that have already been tested.

For each research idea, preserve:

```text
idea
source
why_selected
evidence
hypothesis
expected_effect
```

The system, not the LLM, should assign fields such as:

```text
id
status
```

This prevents the model from incorrectly deciding that an experiment has already been tested.

### Research model

Each executable uses one provider for every LLM node:

```text
all_claude.py → Claude CLI
all_codex.py  → Codex CLI
```

Both use live web search, read-only research access, and structured JSON output.
They use the existing CLI subscription login rather than a separate API key.

### Research search

Planner and Modify Code do not search for papers or invent research sources.

Web/paper retrieval is performed by the selected CLI Researcher using live web
search. Search stays inside the Researcher node; there is no separate Search
node in the first version.

---

## Experiment Planner

Purpose:

> Convert a research idea into a concrete experiment plan.

It does **not** invent a new research direction and does **not** write code.

It receives the single idea selected by Researcher in:

```text
current_idea
```

After an experiment is recorded, the next idea is selected afresh:

```text
Record Result → Researcher → one new current_idea
```

The Planner should inspect the current editable code so the plan matches the actual implementation.

Its output should contain approximately:

```text
idea_id
goal
changes
scope
reasoning
```

Where:

- `goal` = what the experiment tests
- `changes` = concrete modifications required
- `scope` = `small` normally; `structural` only when genuinely required
- `reasoning` = why those modifications test the hypothesis

The Planner uses the provider selected by the executable. It receives the
selected idea and current editable code, but no tools; its only job is to return
a small structured experiment plan.

---

## Modify Code

Purpose:

> Implement the Experiment Planner's plan.

Modify Code must **not**:

- invent a new research direction
- replace the Planner's hypothesis
- make unrelated improvements
- modify read-only files

Initially it modifies only:

```text
train.py
```

Before modifying the file, enough information must be preserved to allow a revert.

Current simple approach:

```text
save previous train.py content in State
save an exact snapshot under memory/code_snapshots/
```

The snapshot is written by deterministic Python code before the candidate is
written. The LLM never recreates the old file. If the experiment is rejected or
fails, Revert restores this exact snapshot. If it improves `val_bpb`, the edited
`train.py` remains the new accepted version. Snapshots are retained locally for
inspection and can later be replaced by a Git-based keep/revert workflow.

Modify Code should also produce metadata describing:

```text
what changed
why it changed
```

This information will later be written into the experiment history.

Coding model:

```text
Claude CLI or Codex CLI
selected once for the whole run
structured JSON output
```

The selected model does not return a complete replacement file. A normal
`small` experiment allows at most 3 edits, 20 replaced existing lines per edit,
and 30 introduced lines per edit. Planner may select `structural` only when the
hypothesis truly requires it; that controlled scope allows at most 5 edits, 80
replaced lines and 80 introduced lines per edit. Program logic still rejects
invalid ranges, overlaps, exceeded budgets, and resulting Python syntax errors.

---

## Run Experiment

Purpose:

> Execute the modified training code.

This node is deterministic and does not need an LLM.

It should run `train.py` using the same Python environment as the LangGraph process.

Preferred execution approach:

```text
sys.executable + subprocess
```

It should capture at least:

```text
stdout
stderr
return code
timeout status
log file path
```

Run Experiment does **not** interpret `val_bpb`.

It writes complete combined stdout/stderr to `memory/runs/`. State contains only
the log path, a short summary, and the final 50 lines for a failed run.

---

## Read Metric

Purpose:

> Extract `val_bpb` from the experiment output.

This node should not decide whether the experiment was good or bad.

It reads the run log and extracts both `val_bpb` and `peak_memory_mb`.

---

## Evaluate

Purpose:

> Compare the new experiment result against the best known result.

Main objective:

```text
lower val_bpb = better
```

The current rule remains `current_score < best_score`. Peak memory and changed
line counts are recorded now, but will become decision criteria only after real
Apple runs establish useful thresholds.

---

## Repair Code

Purpose:

> Give a simple runtime implementation error one repair attempt.

Timeout and recognizable OOM/killed failures go directly to Revert. Other
runtime crashes receive at most one small repair attempt. Repair sees the final
50 log lines and must preserve the same idea and plan. If repair fails or the
rerun crashes, the original snapshot is restored and the crash is recorded.

---

## Keep

Purpose:

> Preserve a successful modification.

The eventual preferred mechanism is likely Git-based.

Current decision: keep the deterministic snapshot mechanism. Git commit/reset
was evaluated but is not implemented because this working directory is not yet
a Git repository. If Git is adopted later, setup must first define a dedicated
experiment branch and clean-worktree policy; commits must include only the
experiment's `train.py` change so unrelated user work can never be reset.

---

## Revert

Purpose:

> Restore the previous successful version after a failed experiment.

Initial implementation may restore the saved original source code.

Git reset remains a later option after the repository and branch safety rules
above exist. Snapshot restore is the active mechanism.

---

## Record Result

Purpose:

> Persist everything learned from the experiment.

After recording, the graph always returns to Researcher unless the requested
experiment limit has been reached.

---

# 5. Code Permissions

Initial permissions:

## Editable

```text
train.py
```

Meaning:

```text
read + write
```

## Read-only

```text
prepare.py
```

Meaning:

```text
read only
```

Researcher and Planner may inspect read-only code if necessary for understanding the system.

Modify Code must never write to it.

---

# 6. Constraints

Initial constraints:

- `prepare.py` cannot be modified.
- Dataset cannot be modified.
- Evaluation metric cannot be modified.
- Experiments should remain comparable.
- Do not casually introduce new dependencies.
- All real experiments must use the same experiment budget.
- Research direction should come from Researcher, not Modify Code.
- Implementation planning should come from Planner, not Modify Code.

These constraints can be revised deliberately later, but agents should not change them themselves.

## Upstream training files

The project started from `train.py` and `prepare.py` in
`karpathy/autoresearch` commit:

```text
228791fb499afffb54b46200aca536f79142f117
```

`prepare.py` remains unchanged. `train.py` is now the Apple Silicon port: it
uses MPS, PyTorch scaled-dot-product attention, eager FP32 execution, an
MPS-sized training batch, and MPS synchronization/memory reporting. The fixed
BPB formula and `EVAL_TOKENS` amount remain unchanged. Apple validation uses a
separate measured `EVAL_BATCH_SIZE = 8`; this changes throughput, not the metric
or the number of evaluated tokens. Apple results should be
compared with other runs from the same Apple baseline, not the upstream H100
numbers.

---

# 7. State vs Persistent Memory

Important distinction:

```text
LangGraph State
= current working memory

Persistent files
= long-term research memory
```

State should not be treated as the only memory system.

---

# 8. Persistent Memory

Initial persistence format:

```text
memory/
├── research.jsonl
├── experiments.jsonl
├── code_snapshots/
└── runs/
```

JSONL was chosen because it is:

- simple
- human-readable
- easy to inspect in VS Code
- easy to append to
- sufficient for the first version

SQLite is unnecessary initially.

---

## research.jsonl

Stores Researcher's findings.

A research record should preserve approximately:

```text
id
idea
source
why_selected
evidence
hypothesis
expected_effect
status
```

The goal is that a human can inspect this file and understand:

> What did the system find, where did it find it, and why did it think this was worth testing?

---

## experiments.jsonl

Stores experiments that were actually executed.

Records should eventually contain approximately:

```text
idea_id
hypothesis
experiment plan
code changes
change reasoning
added/removed line counts
repair attempt count
previous score
new score
peak memory
keep/revert decision
execution status
run log paths
commit hash / version information
```

This allows Researcher to distinguish:

```text
things previously discovered

vs.

things actually tested
```

---

# 9. State

Current intended State contains operational information such as:

```text
editable_files
read_only_files
constraints

research_history

current_idea
experiment_plan

original_code
code_changes
code_change_reasoning

experiment_log_path
experiment_summary
experiment_error_tail
experiment_returncode
repair_attempts

current_score
best_score
peak_memory_mb
```

Not every field needs to exist from the first line of the program.

Optional fields can be represented with `NotRequired`.

---

# 10. Research Idea Structure

A persisted research idea should approximately follow:

```text
id
idea
source
why_selected
evidence
hypothesis
expected_effect
status
```

Possible statuses currently considered:

```text
selected
tested
rejected
```

Older persisted records may still contain the legacy `pending` value; they are
kept as research history but no longer form a queue.

The LLM should primarily generate the semantic research content.

Program logic should manage system metadata such as IDs and lifecycle status.

---

# 11. Structured LLM Output

Researcher, Planner and Modify Code should preferably use structured outputs rather than relying on arbitrary free-form model responses.

Conceptually:

```text
Researcher
→ Research idea schema

Planner
→ Experiment plan schema

Modify Code
→ Code modification schema
```

Two model-consistent versions are available:

```text
all_claude.py → Researcher + Planner + Modify Code use Claude CLI
all_codex.py  → Researcher + Planner + Modify Code use Codex CLI
```

Everything else—prompts, schemas, graph routing, memory, validation,
keep/revert, and training—is shared. Researcher receives read/search tools;
Planner and Modify Code receive no tools.

---

# 12. Experiment Budget

The M4 MacBook protocol defines:

```text
TIME_BUDGET = 1800 seconds
```

The original five-minute protocol over-rewarded changes that improved early
throughput or very early learning. Thirty minutes is long enough to make the
training trajectory more meaningful on this M4 MacBook while remaining usable
for local experiments. The subprocess timeout is the training budget plus one
hour so the fixed full validation can finish. A baseline must be run before
autonomous experiments begin.

The `baseline` command records its time budget, evaluation token count, score,
peak memory, status, and run log in `experiments.jsonl`. The `run` command reads
the most recent successful `keep` score only from the same time/evaluation
protocol, so five-minute and thirty-minute scores are never compared directly.
Researcher and Planner also receive the current protocol explicitly. Legacy
results remain available as lessons, but not as direct score comparisons.

The Apple port uses a smaller fixed training batch appropriate for MPS while
preserving the original fixed evaluation token count. Every comparable Apple
experiment uses this same setup.

---

# 13. API / Cost Constraint

The Claude version consumes the user's authenticated Claude CLI plan quota. The
Codex version consumes the user's authenticated Codex CLI plan quota. Neither
requires a separately configured model API key. The selected CLI must be logged
in before its version is run.

---

# 14. Design Principle: Stay Close to AutoResearch

Do not add complexity merely because LangGraph supports it.

Avoid unnecessary:

- multi-agent hierarchies
- databases
- elaborate memory frameworks
- reviewer/critic agents
- orchestration layers

unless experiments later demonstrate a concrete need.

The intended first version is:

```text
simple
observable
local
cheap
easy to understand
close to Karpathy's original research loop
```

---

# 15. Open Decisions

These are **not decided yet**:

- Exact strategy for selecting the single next idea
- Exact duplicate-detection mechanism
- Exact improvement threshold/tie handling
- Final Keep/Revert Git strategy
- Commit strategy
- Whether `prepare.py` content always needs to be shown to Researcher
- Whether Planner needs access to read-only source code
- Whether each CLI's default subscription model should later be pinned explicitly
- When/if JSONL should eventually become SQLite

---

# 16. Important Implementation Note

This document records **design decisions**, not a guarantee that every discussed code snippet has already been added to the current VS Code project.

When continuing implementation, first inspect the actual current source code and reconcile it with this document rather than assuming a previously discussed snippet exists.
