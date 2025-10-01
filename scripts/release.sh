#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/config/app.yml"
SHADOW_ID=""
SKIP_VALIDATIONS=0
DISABLE_ROLLOUT=0

usage() {
  cat <<USAGE
Usage: $(basename "$0") --shadow-id MODEL_ID [--config PATH] [--skip-validations] [--disable-rollout]

Orchestrates the rollout flow:
  1. Run validations (config + smoke tests).
  2. Promote candidate to shadow slot.
  3. Flip rollout.mode to canary.
  4. Wait for operator confirmation after monitoring.
  5. Promote to active and ramp to full (or disable rollout when requested).

Options:
  --shadow-id MODEL_ID   Registry model identifier to promote.
  --config PATH          Override path to config/app.yml (default: config/app.yml).
  --skip-validations     Skip validation step (reuse existing results).
  --disable-rollout      Leave rollout disabled after promotion instead of setting mode=full.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shadow-id)
      SHADOW_ID="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$(cd "$ROOT_DIR" && python - <<'PY' "$2"
import sys
from pathlib import Path
path = Path(sys.argv[1])
print(path if path.is_absolute() else Path.cwd() / path)
PY
)"
      shift 2
      ;;
    --skip-validations)
      SKIP_VALIDATIONS=1
      shift
      ;;
    --disable-rollout)
      DISABLE_ROLLOUT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$SHADOW_ID" ]]; then
  echo "[release] --shadow-id is required" >&2
  usage
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[release] config not found at $CONFIG_PATH" >&2
  exit 1
fi

update_rollout_mode() {
  local mode="$1"
  python - "$CONFIG_PATH" "$mode" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
mode = sys.argv[2]
config = yaml.safe_load(config_path.read_text())
config.setdefault("rollout", {})["mode"] = mode
config_path.write_text(yaml.safe_dump(config, sort_keys=False))
print(f"[release] rollout.mode -> {mode}")
PY
}

initial_mode=$(python - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path
import yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text())
print(config.get("rollout", {}).get("mode", "disabled"))
PY
)

cleanup() {
  update_rollout_mode "$initial_mode" >/dev/null 2>&1 || true
}

trap cleanup ERR

run_validations() {
  echo "[release] running config validation"
  (cd "$ROOT_DIR" && PYTHONPATH=. python scripts/validate_config.py --config "$CONFIG_PATH")
  if [[ -f "$ROOT_DIR/scripts/quality_gate.sh" ]]; then
    echo "[release] running quality gate"
    (cd "$ROOT_DIR" && bash "$ROOT_DIR/scripts/quality_gate.sh")
  else
    echo "[release] quality gate script not found, skipping"
  fi
}

if [[ "$SKIP_VALIDATIONS" -eq 0 ]]; then
  run_validations
else
  echo "[release] validations skipped by operator"
fi

echo "[release] promoting $SHADOW_ID to shadow"
(cd "$ROOT_DIR" && PYTHONPATH=. python scripts/promote_model.py --to shadow --id "$SHADOW_ID")

if [[ "$initial_mode" != "canary" ]]; then
  update_rollout_mode "canary"
else
  echo "[release] rollout already in canary mode"
fi

echo "[release] restart ai_scorer/executor with updated config, then monitor Grafana (winrate, markout, calibration, drift)"
read -r -p "[release] Continue to promote to active? (y/N): " answer
if [[ ! "$answer" =~ ^[Yy]$ ]]; then
  echo "[release] aborting before active promotion"
  update_rollout_mode "$initial_mode"
  exit 0
fi

echo "[release] promoting $SHADOW_ID to active"
(cd "$ROOT_DIR" && PYTHONPATH=. python scripts/promote_model.py --to active --id "$SHADOW_ID")

if [[ "$DISABLE_ROLLOUT" -eq 1 ]]; then
  update_rollout_mode "disabled"
else
  update_rollout_mode "full"
fi

echo "[release] release complete"
