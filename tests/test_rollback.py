from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from app.common.config import RolloutGuardConfig
from app.risk.main import RollbackManager


def _write_model(dir_path: Path, model_id: str, created_at: str) -> dict[str, str]:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "model.json").write_text(json.dumps({"id": model_id}))
    (dir_path / "model.onnx").write_bytes(b"onnx")
    return {"id": model_id, "created_at": created_at, "path": dir_path.name}


def _prepare_registry(tmp_path: Path, model_ids: list[str]) -> Path:
    registry = tmp_path / "registry"
    registry.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, model_id in enumerate(model_ids, start=1):
        manifest.append(_write_model(registry / model_id, model_id, f"2025-01-{idx:02d}T00:00:00Z"))
    (registry / "manifest.json").write_text(json.dumps(manifest))
    shutil.copytree(registry / model_ids[-1], registry / "active")
    return registry


def _prepare_config(tmp_path: Path, mode: str = "canary") -> Path:
    config_path = tmp_path / "app.yml"
    config_path.write_text(
        yaml.safe_dump({"rollout": {"mode": mode}}, sort_keys=False)
    )
    return config_path


def test_winrate_degradation_triggers_rollback(tmp_path):
    registry = _prepare_registry(tmp_path, ["model_a", "model_b"])
    config_path = _prepare_config(tmp_path, mode="canary")
    guard_cfg = RolloutGuardConfig(
        winrate_delta_pp=5,
        min_sample=10,
        psi_block=0.3,
        ks_block=0.2,
        cooldown_minutes=15,
    )
    manager = RollbackManager(
        guard_cfg=guard_cfg,
        config_path=config_path,
        registry_root=registry,
        initial_mode="canary",
    )
    result = manager.update_winrates(baseline=0.60, canary=0.40, sample=50, now_ms=0)
    assert result and result.executed
    active_meta = json.loads((registry / "active" / "model.json").read_text())
    assert active_meta["id"] == "model_a"
    raw_cfg = yaml.safe_load(config_path.read_text())
    assert raw_cfg["rollout"]["mode"] == "disabled"


def test_parity_fail_triggers_rollback(tmp_path):
    registry = _prepare_registry(tmp_path, ["model_a", "model_b"])
    config_path = _prepare_config(tmp_path, mode="canary")
    guard_cfg = RolloutGuardConfig()
    manager = RollbackManager(
        guard_cfg=guard_cfg,
        config_path=config_path,
        registry_root=registry,
        initial_mode="canary",
    )
    assert manager.record_parity_metric(0.0, {"stage": "online"}, now_ms=0) is None
    result = manager.record_parity_metric(1.0, {"stage": "online"}, now_ms=1_000)
    assert result and result.executed


def test_cooldown_blocks_duplicate_rollbacks(tmp_path):
    registry = _prepare_registry(tmp_path, ["model_a", "model_b", "model_c"])
    config_path = _prepare_config(tmp_path, mode="canary")
    guard_cfg = RolloutGuardConfig(cooldown_minutes=15, psi_block=0.3)
    manager = RollbackManager(
        guard_cfg=guard_cfg,
        config_path=config_path,
        registry_root=registry,
        initial_mode="canary",
    )
    first = manager.record_psi_metric("psi_feature_q", 0.5, now_ms=0)
    assert first and first.executed
    active_after_first = json.loads((registry / "active" / "model.json").read_text())["id"]
    assert active_after_first == "model_b"
    second = manager.record_psi_metric("psi_feature_q", 0.6, now_ms=1)
    assert second and not second.executed
    assert second.skipped == "cooldown"
    active_after_second = json.loads((registry / "active" / "model.json").read_text())["id"]
    assert active_after_second == "model_b"
