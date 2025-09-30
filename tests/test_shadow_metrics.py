from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ai_scorer import main as ai_main
from app.common.schema import Candidate, EntryRef, Side


class DummyRedis:
    def __init__(self) -> None:
        self.metrics = []

    def xadd(self, stream, metric, idempotency_key=None, maxlen=None):  # noqa: D401 signature kept
        if stream == "metrics:ai":
            self.metrics.append(metric)
        return None


def build_service() -> ai_main.AIScorerService:
    svc = ai_main.AIScorerService.__new__(ai_main.AIScorerService)
    svc.rs = DummyRedis()
    svc.models = {}
    svc.active_model_name = "active"
    svc.recent_scores = ai_main.RunningStats()
    svc.score_metric_interval = 10 ** 6
    svc.score_metric_counter = 0
    svc._emit_score_metrics = lambda: None
    svc.drift_monitor = SimpleNamespace(update=lambda *args, **kwargs: [])
    svc._emit_metrics = lambda items: None
    svc.pending = defaultdict(deque)
    svc.calibration_monitor = None
    svc.shadow_metrics = None
    svc._apply_spread_entry = lambda base, _feats: base
    svc._predict = lambda handle, vector, fallback: 0.7 if handle.model_id == "active_v1" else 0.65
    svc.cfg = SimpleNamespace(
        ai_scorer=SimpleNamespace(
            thresholds=SimpleNamespace(entry=0.6, hold=0.4)
        )
    )
    active_handle = ai_main.ModelHandle(
        role="active",
        model_id="active_v1",
        feature_names=["f1", "f2"],
        session=None,
        input_name=None,
        prob_idx=None,
        calibrator=ai_main.ProbabilityCalibrator(None, None),
        thresholds={"tau_entry": 0.6, "tau_hold": 0.4},
        metadata={},
        path=Path("."),
    )
    shadow_handle = ai_main.ModelHandle(
        role="shadow",
        model_id="shadow_v1",
        feature_names=["f1", "f2"],
        session=None,
        input_name=None,
        prob_idx=None,
        calibrator=ai_main.ProbabilityCalibrator(None, None),
        thresholds={"tau_entry": 0.55, "tau_hold": 0.35},
        metadata={},
        path=Path("."),
    )
    svc.models = {"active": active_handle, "shadow": shadow_handle}
    svc.active_handle = active_handle
    svc.shadow_handle = shadow_handle
    return svc


def make_candidate(score: float = 0.7) -> Candidate:
    entry = EntryRef(type="limit")
    return Candidate(
        ts=1,
        symbol="BTCUSDT",
        side=Side.BUY,
        entry_ref=entry,
        tp_pct=0.01,
        sl_pct=0.005,
        features={"f1": 0.1, "f2": 0.2},
        score=score,
    )


def find_metric(rs: DummyRedis, name: str):
    return [metric for metric in rs.metrics if metric.name == name]


def test_shadow_metrics_emitted():
    svc = build_service()
    cand = make_candidate()
    svc._score_candidate(cand)

    eval_metrics = find_metric(svc.rs, "shadow_signals_evaluated_total")
    assert eval_metrics, "shadow signals evaluated metric missing"
    approved_metrics = find_metric(svc.rs, "shadow_signals_approved_total")
    assert approved_metrics, "shadow signals approved metric missing"
    dist_metrics = find_metric(svc.rs, "shadow_score_distribution")
    assert dist_metrics and dist_metrics[0].labels.get("split") == "online"
    record = svc.pending[cand.symbol][-1]
    assert record.shadow_score is not None


def test_tau_metric_and_model_ids():
    svc = build_service()
    svc._emit_tau_metric("active", svc.active_handle.thresholds)
    svc._emit_tau_metric("shadow", svc.shadow_handle.thresholds)
    svc._emit_model_identifiers()

    tau_metrics = find_metric(svc.rs, "tau_entry_current")
    assert any(m.labels.get("model") == "active" for m in tau_metrics)
    assert any(m.labels.get("model") == "shadow" for m in tau_metrics)

    active_ids = find_metric(svc.rs, "model_active_id")
    shadow_ids = find_metric(svc.rs, "model_shadow_id")
    assert active_ids and active_ids[0].labels.get("id") == "active_v1"
    assert shadow_ids and shadow_ids[0].labels.get("id") == "shadow_v1"
