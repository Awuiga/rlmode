#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
pip install pre-commit
pre-commit install
pre-commit run --all-files
echo "pre-commit installed and ran successfully."
