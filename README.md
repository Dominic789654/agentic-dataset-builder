# Agentic Dataset Builder

Build one merged training dataset from local Pi and Codex session history.

## Setup

Use Python 3.10+.

Recommended one-shot environment setup:

```bash
./setup.sh
source .venv/bin/activate
```

Manual alternative:

```bash
pip install -r requirements.txt
```

## What users run

From this directory:

```bash
./run.sh --output-root ./out
```

That one command will:

- scan `~/.pi/agent/sessions`
- scan `~/.codex/sessions`
- convert session history into the local Qwen3.5 schema
- label records as `cot_eligible`, `agent_only`, or `discard`
- keep `cot_eligible` and `agent_only`
- merge them into one final parquet file
- remove intermediate directories automatically after success

## Final output

Each run creates one directory under `./out/`:

```text
out/agentic-dataset-<timestamp>/
  dataset.parquet
  manifest.json
  run.log
```

Default deliverable is just one user-facing dataset file:

- `dataset.parquet`

Supporting files:

- `manifest.json`: what was scanned, what was kept, summary stats
- `run.log`: full step-by-step execution log

## Common options

```bash
# only Pi
./run.sh --output-root ./out --include-sources pi

# only Codex
./run.sh --output-root ./out --include-sources codex

# keep intermediates for debugging
./run.sh --output-root ./out --keep-intermediates

# also emit final merged jsonl/jsonl.gz
./run.sh --output-root ./out --final-format both
```

## What is kept by default

- `cot_eligible`: agentic traces with visible reasoning
- `agent_only`: agentic traces without visible reasoning

`discard` records are excluded from the final dataset by default.

## Package layout

```text
agentic_dataset/
  build_agentic_dataset.py
  export_pi_session.py
  export_pi_session_to_qwen35.py
  export_codex_session_to_qwen35.py
  label_qwen35_agentic.py
  export_qwen35_training.py
  qwen35_training_record.py
run.sh
README.md
```

## Notes

- Pi currently provides much better visible reasoning coverage than Codex.
- Codex traces are still useful for agent-behavior distillation even when reasoning is encrypted-only.
- Redaction is not included yet. Add it before distributing the tool broadly if users may have sensitive local data.
