#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


REGISTRY_ROOT = Path(os.environ.get("MODEL_REGISTRY_ROOT", "models/registry"))
MANIFEST_PATH = REGISTRY_ROOT / "manifest.json"


def load_manifest() -> Dict[str, Dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        data = json.loads(MANIFEST_PATH.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid manifest.json: {exc}")
    result: Dict[str, Dict[str, Any]] = {}
    for entry in data or []:
        model_id = entry.get("id")
        if model_id:
            result[str(model_id)] = entry
    return result


def ensure_link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    try:
        dst.symlink_to(src, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def promote_model(target: str, model_id: str) -> None:
    if target not in {"active", "shadow"}:
        raise SystemExit("--to must be 'active' or 'shadow'")
    registry = REGISTRY_ROOT
    model_dir = registry / model_id
    if not model_dir.exists():
        raise SystemExit(f"model directory not found: {model_dir}")
    model_file = model_dir / "model.onnx"
    meta_file = model_dir / "model.json"
    if not model_file.exists() or not meta_file.exists():
        raise SystemExit("model.onnx and model.json must exist in the model directory")
    manifest = load_manifest()
    if manifest and model_id not in manifest:
        raise SystemExit(f"model id '{model_id}' not present in manifest.json")
    ensure_link(model_dir, registry / target)
    print(f"promoted {model_id} -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a model to active or shadow slot")
    parser.add_argument("--to", required=True, help="Slot to promote to: active | shadow")
    parser.add_argument("--id", required=True, help="Model identifier present in registry")
    args = parser.parse_args()
    promote_model(args.to, args.id)


if __name__ == "__main__":
    main()
