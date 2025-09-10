#!/usr/bin/env bash
set -euo pipefail

echo "[quality] formatting and linting..."
if command -v ruff >/dev/null 2>&1; then
  ruff check .
fi
if command -v black >/dev/null 2>&1; then
  black --check .
fi
echo "[quality] type-check..."
if command -v mypy >/dev/null 2>&1; then
  mypy dump_market_to_parquet.py app/features/core.py
fi
echo "[quality] tests..."
pytest -q

