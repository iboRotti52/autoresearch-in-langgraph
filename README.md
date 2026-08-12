# AutoResearch in LangGraph

A small, understandable LangGraph reimplementation of the core AutoResearch
loop, adapted for Apple Silicon with PyTorch MPS.

The project has two interchangeable versions:

- `all_codex.py`: Researcher, Planner, Modify Code, and Repair use Codex CLI.
- `all_claude.py`: the same nodes use Claude CLI.

The graph keeps the training and evaluation logic shared. It automatically
creates a baseline, proposes one experiment at a time, edits only `train.py`,
runs the experiment, reads `val_bpb`, and keeps or reverts the change.

## Requirements

- Apple Silicon Mac with MPS support
- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- A logged-in Codex CLI or Claude CLI subscription

## Setup

```bash
uv sync
uv run prepare.py
```

The preparation step downloads 10 training shards plus the pinned validation
shard and builds an 8,192-token tokenizer under `~/.cache/autoresearch`.

## Run with Codex

Create the baseline first:

```bash
uv run all_codex.py baseline
```

Then run one autonomous experiment:

```bash
uv run all_codex.py run --max-experiments 1
```

Use `all_claude.py` instead for the all-Claude version.

## Current Apple protocol

- Training budget: 1,800 seconds (30 minutes)
- Evaluation: 20,971,520 fixed validation tokens
- Training batch: 1 MPS sequence with gradient accumulation
- Validation batch: 8 MPS sequences
- Metric: validation bits per byte (`val_bpb`, lower is better)

Scores are compared only when their training budget and evaluation token count
match. The older five-minute results remain in memory as research lessons but
are not used as the best score for the current protocol.

## Memory

- `memory/research.jsonl`: research ideas and their status
- `memory/experiments.jsonl`: baseline and experiment results
- `memory/runs/`: full stdout logs, ignored by Git
- `memory/code_snapshots/`: pre-experiment snapshots, ignored by Git

See [`design.md`](design.md) for the complete node-by-node design.
