# AutoResearch in LangGraph

This is my small LangGraph implementation inspired by
[Andrej Karpathy's AutoResearch](https://github.com/karpathy/autoresearch),
adapted to run on Apple Silicon with PyTorch MPS.

I built it as a learning project. Instead of asking an LLM to create a complete
autonomous system at once, I wanted to understand the research loop step by
step: where an idea comes from, how it becomes an experiment, what code changes,
and why a result is kept or reverted.

This is not an official Karpathy project or a line-for-line port.

## How it works

```text
Researcher → Planner → Modify Code → Run Experiment
           → Read Metric → Keep/Revert → Record Result
           → Researcher chooses one new idea
```

- **Researcher** reads the code and previous results, then selects one idea. It
  can search papers but cannot edit code.
- **Planner** turns that idea into the smallest concrete experiment.
- **Modify Code** implements only the plan and may edit only `train.py`.
- The program runs training, reads `val_bpb`, and uses deterministic logic to
  keep an improvement or restore the previous snapshot.
- Ideas and results are stored in JSONL so later experiments can learn from the
  history.

## What comes from AutoResearch

I kept the core loop: `prepare.py` stays fixed, the agent edits `train.py`, each
experiment has a fixed time budget, lower `val_bpb` wins, and worse changes are
reverted.

The main difference is that the responsibilities are explicit LangGraph nodes.
I also added structured outputs, persistent research memory, controlled edit
limits, snapshots, and one small repair attempt for simple crashes.

There are two versions using the same graph:

- `all_codex.py` uses the Codex CLI subscription.
- `all_claude.py` uses the Claude CLI subscription.

## Apple Silicon notes

Karpathy's original setup targets an NVIDIA GPU. I adapted the training code for
my M4 MacBook with PyTorch MPS.

The complete loop was tested with a five-minute run. That test improved
`val_bpb` from `2.034278` to `1.961818`, but it also showed that very short runs
can favor changes that mainly help early training. The current protocol therefore
uses a 30-minute training budget. Old five-minute results remain in history but
are not compared directly with the new protocol.

The 30-minute configuration and protocol logic are verified, but a complete
30-minute baseline and experiment have not been run yet.

## Setup

Requirements: Apple Silicon with MPS, Python 3.10+, `uv`, and a logged-in Codex
CLI or Claude CLI subscription.

```bash
uv sync
uv run prepare.py
```

The preparation step downloads 10 training shards, one fixed validation shard,
and creates the tokenizer under `~/.cache/autoresearch`.

## Run

Create the baseline first:

```bash
uv run all_codex.py baseline
```

Then run one autonomous experiment:

```bash
uv run all_codex.py run --max-experiments 1
```

Use `all_claude.py` instead for the Claude version.

The current Apple protocol trains for 1,800 seconds and evaluates on a fixed
20,971,520 validation tokens. Full logs and code snapshots are kept locally;
research ideas and experiment results are stored under `memory/`.

See [`design.md`](design.md) for the detailed node-by-node design.
