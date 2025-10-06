from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.ai_scorer import main as ai_main


class DummySession:
    def run(self, *_args, **_kwargs):
        return [np.zeros((8, 1), dtype=np.float32)]


def setup_registry(tmp_path: Path, models: dict[str, dict[str, object]]) -> None:
    registry = tmp_path / "registry"
    registry.mkdir(parents=True, exist_ok=True)
    manifest = []
    for model_id, meta in models.items():
        model_dir = registry / model_id
        model_dir.mkdir()
        (model_dir / "model.onnx").write_bytes(b"fake-onnx")
        (model_dir / "model.json").write_text(json.dumps(meta))
        manifest.append({
            "id": model_id,
            "created_at": meta.get("created_at", "2025-01-01T00:00:00Z"),
            "schema_hash": meta.get("schema_hash"),
            "thresholds": meta.get("thresholds", {}),
            "path": model_id,
            "notes": meta.get("notes", ""),
        })
    (registry / "manifest.json").write_text(json.dumps(manifest))
    ai_main.REGISTRY_ROOT = registry


def dummy_feature_state(schema_hash: str, feature_names: list[str]) -> object:
    return SimpleNamespace(
        config_hash=schema_hash,
        field_order=lambda: list(feature_names),
    )


@pytest.fixture(autouse=True)
def patch_onnx(monkeypatch):
    monkeypatch.setenv("MODEL_REGISTRY_ROOT", "")
    monkeypatch.setattr(ai_main, "try_load_onnx", lambda path: (DummySession(), "input", 0))
    yield


def test_shadow_schema_mismatch(monkeypatch, tmp_path):
    setup_registry(
        tmp_path,
        {
            "active_v1": {
                "id": "active_v1",
                "schema_hash": "hash-active",
                "thresholds": {"tau_entry": 0.6, "tau_hold": 0.4},
                "feature_names": ["f1", "f2"],
            },
            "shadow_v1": {
                "id": "shadow_v1",
                "schema_hash": "mismatch",
                "thresholds": {"tau_entry": 0.55, "tau_hold": 0.35},
                "feature_names": ["f1", "f2"],
            },
        },
    )
    feature_state = dummy_feature_state("hash-active", ["f1", "f2"])
    handle = ai_main.load_registry_model(
        "active",
        feature_state,
        calibration_enabled=True,
        default_thresholds={"tau_entry": 0.6, "tau_hold": 0.4},
        allow_schema_mismatch=False,
    )
    assert handle.model_id == "active_v1"
    shadow = ai_main.load_registry_model(
        "shadow",
        feature_state,
        calibration_enabled=True,
        default_thresholds={"tau_entry": 0.6, "tau_hold": 0.4},
        allow_schema_mismatch=True,
        shadow_cfg=SimpleNamespace(use_threshold_from_meta=True, tau_entry=None, tau_hold=None),
    )
    assert shadow is None


def test_active_schema_mismatch_fatal(tmp_path):
    setup_registry(
        tmp_path,
        {
            "bad_active": {
                "id": "bad_active",
                "schema_hash": "wrong",
                "thresholds": {"tau_entry": 0.6, "tau_hold": 0.4},
                "feature_names": ["f1"],
            }
        },
    )
    feature_state = dummy_feature_state("expected", ["f1"])
    with pytest.raises(RuntimeError):
        ai_main.load_registry_model(
            "active",
            feature_state,
            calibration_enabled=True,
            default_thresholds={"tau_entry": 0.6, "tau_hold": 0.4},
            allow_schema_mismatch=False,
        )


def test_parity_failure(monkeypatch, tmp_path):
    setup_registry(
        tmp_path,
        {
            "active_v1": {
                "id": "active_v1",
                "schema_hash": "hash",
                "thresholds": {"tau_entry": 0.6, "tau_hold": 0.4},
                "feature_names": ["f1"],
            },
        },
    )
    feature_state = dummy_feature_state("hash", ["f1"])
    monkeypatch.setattr(ai_main, "_parity_self_test", lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError):
        ai_main.load_registry_model(
            "active",
            feature_state,
            calibration_enabled=True,
            default_thresholds={"tau_entry": 0.6, "tau_hold": 0.4},
            allow_schema_mismatch=False,
        )


def test_shadow_threshold_override(monkeypatch, tmp_path):
    setup_registry(
        tmp_path,
        {
            "active_v1": {
                "id": "active_v1",
                "schema_hash": "hash",
                "thresholds": {"tau_entry": 0.6, "tau_hold": 0.4},
                "feature_names": ["f1", "f2"],
            },
            "shadow_v2": {
                "id": "shadow_v2",
                "schema_hash": "hash",
                "thresholds": {"tau_entry": 0.55, "tau_hold": 0.35},
                "feature_names": ["f1", "f2"],
            },
        },
    )
    feature_state = dummy_feature_state("hash", ["f1", "f2"])
    handle = ai_main.load_registry_model(
        "shadow",
        feature_state,
        calibration_enabled=True,
        default_thresholds={"tau_entry": 0.6, "tau_hold": 0.4},
        allow_schema_mismatch=True,
        shadow_cfg=SimpleNamespace(use_threshold_from_meta=False, tau_entry=0.7, tau_hold=0.5),
    )
    assert handle is not None
    assert handle.thresholds["tau_entry"] == pytest.approx(0.7)
    assert handle.thresholds["tau_hold"] == pytest.approx(0.5)
