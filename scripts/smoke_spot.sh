#!/usr/bin/env bash
set -euo pipefail

docker compose up -d redis collector parquet-dumper
echo "[smoke] wait for data…"
sleep 90
FOUND=$(find ./data/parquet -type f -name '*.parquet' | head -1 || true)
if [ -z "$FOUND" ]; then
  echo "[smoke] FAIL: no parquet files found"
  exit 1
else
  echo "[smoke] found $(basename "$FOUND")"
  SIDE="${FOUND%.parquet}.json"
  if [ -f "$SIDE" ]; then
    if python - <<'PY'
import json, os, sys
p=os.environ.get('SIDE','')
try:
    with open(p,'r',encoding='utf-8') as f:
        meta=json.load(f)
    rc=int(meta.get('row_count',0) or 0)
    sys.exit(0 if rc>0 else 1)
except Exception:
    sys.exit(1)
PY
    then
      echo "[smoke] OK: row_count>0 in sidecar"
    else
      echo "[smoke] FAIL: sidecar row_count<=0 or missing"
      exit 1
    fi
  else
    echo "[smoke] WARN: sidecar not found; assuming OK"
  fi
fi
