import math
from types import SimpleNamespace

import numpy as np

from app.ai_scorer.main import CalibrationMonitor, DriftMonitor


def make_drift_cfg():
    feature_cfg = SimpleNamespace(name="depth", bins=[0.0, 0.5, 1.0], psi_alert=0.3, ks_alert=0.2)
    return SimpleNamespace(
        enabled=True,
        window=10,
        emit_interval=10,
        features=[feature_cfg],
        score_bins=[0.0, 0.5, 1.0],
        psi_alert=0.3,
        ks_alert=0.2,
        brier_alert=0.2,
        ece_alert=0.1,
    )


def test_drift_monitor_emits_prefixed_metrics():
    cfg = make_drift_cfg()
    monitoring_meta = {
        "score": {"bins": [0.0, 0.5, 1.0], "reference": [0.5, 0.5]},
        "features": {
            "depth": {"bins": [0.0, 0.5, 1.0], "reference": [0.5, 0.5]},
        },
    }
    monitor = DriftMonitor(cfg, monitoring_meta)
    metrics = []
    for _ in range(10):
        metrics = monitor.update(0.4, {"depth": 0.4})
    assert metrics, "drift monitor should emit metrics once the window fills"
    names = {name for name, _, _ in metrics}
    assert "psi_feature_depth" in names
    assert "ks_feature_depth" in names
    assert "psi_feature" in names
    assert "ks_feature" in names


def test_calibration_monitor_exports_online_metrics():
    bins = np.linspace(0.0, 1.0, 6)
    monitor = CalibrationMonitor(bins, window=25, brier_alert=0.5, ece_alert=0.5)
    metrics = []
    for idx in range(100):
        prob = 0.6 if idx % 2 else 0.4
        outcome = 1.0 if idx % 3 == 0 else 0.0
        metrics = monitor.update(prob, outcome)
    assert metrics, "calibration monitor should emit metrics once the effective window fills"
    as_dict = {name: (value, labels) for name, value, labels in metrics}
    assert "brier_online" in as_dict
    assert "ece_online" in as_dict
    assert math.isfinite(as_dict["brier_online"][0])
    assert as_dict["brier_online"][1] == {"split": "online"}
    assert as_dict["ece_online"][1] == {"split": "online"}
