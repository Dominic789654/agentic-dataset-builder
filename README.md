# Agentic Dataset Builder

Build one merged training dataset from local Pi and Codex session history.

## Runtime

The core implementation is Python. User-facing entrypoints do not require bash.

Requirements:

- Python 3.10+
- Node 18+ if you want to run via `npx`

## Recommended usage

Published npm-style entrypoint:

```bash
npx agentic-dataset-builder --output-root ./out
```

Local repo usage without bash:

```bash
node cli.mjs --output-root ./out
```

Direct Python entrypoint:

```bash
python run.py --output-root ./out
```

If you want to pre-create the Python environment yourself:

```bash
pip install -r requirements.txt
python run.py --output-root ./out
```

## What users run

From this directory, the simplest no-bash local command is:

```bash
node cli.mjs --output-root ./out
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
node cli.mjs --output-root ./out --include-sources pi

# only Codex
node cli.mjs --output-root ./out --include-sources codex

# keep intermediates for debugging
node cli.mjs --output-root ./out --keep-intermediates

# also emit final merged jsonl/jsonl.gz
node cli.mjs --output-root ./out --final-format both
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
run.py
cli.mjs
README.md
```

## Notes

- default session roots are auto-detected for Linux, macOS, and Windows
- override session paths with `--pi-root` and `--codex-root` if needed
- Pi currently provides much better visible reasoning coverage than Codex.
- Codex traces are still useful for agent-behavior distillation even when reasoning is encrypted-only.
- Redaction is not included yet. Add it before distributing the tool broadly if users may have sensitive local data.
