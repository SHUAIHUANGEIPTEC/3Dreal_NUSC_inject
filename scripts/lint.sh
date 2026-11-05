#!/usr/bin/env bash
# Code hygiene gate: ruff + pyflakes + pytest + hardcoded paths + docs.
#
# Run inside the shared conda env so pytest imports resolve:
#   conda activate nusc-inject && bash scripts/lint.sh
# or pick the interpreter explicitly:
#   PYTHON=~/miniforge3/envs/nusc-inject/bin/python bash scripts/lint.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY=python3
fi

echo "== ruff =="
uvx ruff check src scripts tests

echo "== pyflakes =="
uvx --from pyflakes pyflakes src scripts

echo "== pytest =="
"$PY" -m pytest -q

echo "== hardcoded machine paths =="
matches=$(grep -rnE '/(home|Users|mnt)/' src scripts configs tests \
  --include='*.py' --include='*.yaml' 2>/dev/null | grep -v '__pycache__' || true)
if [ -n "$matches" ]; then
  echo "$matches"
  echo "FAIL: machine-specific path in code/config (moved it to local.yaml?)"
  exit 1
fi

echo "== docs =="
"$PY" scripts/check_docs.py

echo "ALL CLEAN"
