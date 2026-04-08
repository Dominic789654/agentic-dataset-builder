#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Environment ready: $ROOT_DIR/$VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run builder with: ./run.sh --output-root ./out"
