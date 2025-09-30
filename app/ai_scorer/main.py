from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score

from ..common.config import load_app_config
from ..common.logging import setup_logging, get_logger
from ..common.redis_stream import RedisStream
from ..common.schema import Candidate, ApprovedSignal, Metric, Fill
from ..common.utils import utc_ms, gen_id
from ..features.core import FeatureState
from .window_buffer import WindowBuffer

log = get_logger("ai_scorer")



REGISTRY_ROOT = Path(os.environ.get("MODEL_REGISTRY_ROOT", "models/registry"))


def _registry_slot_path(slot: str) -> Path:
    return REGISTRY_ROOT / slot


def _resolve_registry_dir(slot: str) -> Optional[Path]:
    link = _registry_slot_path(slot)
    if not link.exists():
        return None
    try:
        resolved = link.resolve()
    except FileNotFoundError:
        return None
    return resolved


def _load_model_metadata(meta_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(meta_path.read_text())
    except Exception as exc:
        raise RuntimeError(f"failed to read model metadata {meta_path}: {exc}")


def _get_feature_order(state: Any) -> List[str]:
    if hasattr(state, "field_order"):
        order = state.field_order
        if callable(order):
            try:
                order = order()
            except TypeError:
                order = list(order)
        return list(order)
    return []


def _parity_self_test(handle: "ModelHandle", feature_count: int) -> bool:
    if handle.session is None or handle.input_name is None:
        return False
    batch = np.zeros((8, max(feature_count, 1)), dtype=np.float32)
    try:
        outputs = handle.session.run(None, {handle.input_name: batch})
        prob = extract_probability(outputs, handle.prob_idx or 0)
    except Exception as exc:
        log.warning("onnx_parity_self_test_failed", model=handle.model_id, error=str(exc))
        return False
    arr = np.asarray(prob)
    return np.all(np.isfinite(arr)) and np.all(arr >= -1e-6) and np.all(arr <= 1.0 + 1e-6)


def load_registry_model(
    role: str,
    feature_state: "FeatureState",
    *,
    calibration_enabled: bool,
    default_thresholds: Dict[str, float],
    allow_schema_mismatch: bool,
    shadow_cfg: Optional[Any] = None,
) -> Optional["ModelHandle"]:
    resolved_dir = _resolve_registry_dir(role)
    if resolved_dir is None:
        if role == "shadow":
            log.info("shadow_registry_missing", role=role)
            return None
        raise RuntimeError(f"registry slot '{role}' not found under {REGISTRY_ROOT}")
    model_file = resolved_dir / "model.onnx"
    meta_file = resolved_dir / "model.json"
    if not model_file.exists() or not meta_file.exists():
        raise RuntimeError(f"model.onnx or model.json missing in {resolved_dir}")
    metadata = _load_model_metadata(meta_file)
    model_id = str(metadata.get("id") or resolved_dir.name)
    schema_hash = metadata.get("schema_hash")
    expected_hash = getattr(feature_state, "config_hash", None)
    if schema_hash and expected_hash and str(schema_hash) != str(expected_hash):
        msg = f"schema hash mismatch for {model_id}: expected {expected_hash} got {schema_hash}"
        if allow_schema_mismatch:
            log.warning("shadow_schema_mismatch", model=model_id, expected=expected_hash, got=schema_hash)
            return None
        raise RuntimeError(msg)
    feature_names = metadata.get("feature_names") or _get_feature_order(feature_state)
    if not feature_names:
        raise RuntimeError(f"model {model_id} did not specify feature_names")
    session, input_name, prob_idx = try_load_onnx(str(model_file))
    if session is None or input_name is None:
        if allow_schema_mismatch:
            log.warning("shadow_model_load_failed", model=model_id)
            return None
        raise RuntimeError(f"failed to load ONNX model {model_id}")
    thresholds_meta = metadata.get("thresholds") or {}
    thresholds = {
        "tau_entry": float(thresholds_meta.get("tau_entry", default_thresholds.get("tau_entry", 0.5))),
        "tau_hold": float(thresholds_meta.get("tau_hold", default_thresholds.get("tau_hold", 0.3))),
    }
    if shadow_cfg is not None and getattr(shadow_cfg, "use_threshold_from_meta", True) is False:
        if shadow_cfg.tau_entry is not None:
            thresholds["tau_entry"] = float(shadow_cfg.tau_entry)
        if shadow_cfg.tau_hold is not None:
            thresholds["tau_hold"] = float(shadow_cfg.tau_hold)
    calibrator = ProbabilityCalibrator.from_meta(metadata.get("calibration"), calibration_enabled)
    handle = ModelHandle(
        role=role,
        model_id=model_id,
        feature_names=list(feature_names),
        session=session,
        input_name=input_name,
        prob_idx=prob_idx,
        calibrator=calibrator,
        thresholds=thresholds,
        metadata=metadata,
        path=resolved_dir,
    )
    if not _parity_self_test(handle, len(feature_names)):
        if allow_schema_mismatch:
            log.warning("shadow_parity_failed", model=model_id)
            return None
        raise RuntimeError(f"parity self-test failed for model {model_id}")
    log.info("model_ready", role=role, model_id=model_id, schema_hash=schema_hash, path=str(resolved_dir))
    return handle


@dataclass
class DecisionRecord:
    decision_id: str
    symbol: str
    score: float
    ts_ms: int
    shadow_score: Optional[float] = None
    pnl: Optional[float] = None


@dataclass
class RunningStats:
    count: float = 0.0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1.0
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count <= 1.0:
            return 0.0
        return self.m2 / (self.count - 1.0)

    @property
    def std(self) -> float:
        var = self.variance
        return math.sqrt(var) if var > 0 else 0.0


class DriftFeatureMonitor:
    def __init__(self, name: str, bins, reference, *, window: int, emit_interval: int, psi_alert: float, ks_alert: float) -> None:
        self.name = name
        self.bins = np.array(bins, dtype=float)
        if self.bins.ndim != 1 or self.bins.size < 2:
            raise ValueError(f"invalid bins for {name}")
        self.reference = np.array(reference, dtype=float)
        if self.reference.size != self.bins.size - 1:
            self.reference = np.ones(self.bins.size - 1) / (self.bins.size - 1)
        total = float(self.reference.sum()) or 1.0
        self.reference = np.clip(self.reference / total, 1e-6, None)
        self.window = max(window, 10)
        self.emit_interval = max(emit_interval, 1)
        self.psi_alert = psi_alert
        self.ks_alert = ks_alert
        self.values: Deque[float] = deque(maxlen=self.window)
        self.counter = 0

    def update(self, value: float) -> None:
        self.values.append(float(value))
        self.counter += 1

    def ready(self) -> bool:
        return len(self.values) >= min(self.window, 100) and self.counter % self.emit_interval == 0

    def compute(self) -> tuple[float, float]:
        if not self.values:
            return 0.0, 0.0
        hist, _ = np.histogram(list(self.values), bins=self.bins)
        total = float(hist.sum()) or 1.0
        actual = np.clip(hist / total, 1e-6, None)
        expected = self.reference
        psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        actual_cdf = np.cumsum(actual)
        expected_cdf = np.cumsum(expected)
        ks = float(np.max(np.abs(actual_cdf - expected_cdf)))
        return psi, ks


class DriftMonitor:
    def __init__(self, cfg, monitoring_meta: Dict[str, object] | None) -> None:
        self.enabled = bool(cfg and getattr(cfg, "enabled", False) and monitoring_meta is not None)
        self.score_monitor: DriftFeatureMonitor | None = None
        self.feature_monitors: Dict[str, DriftFeatureMonitor] = {}
        if not self.enabled:
            return
        score_meta = (monitoring_meta or {}).get("score") if monitoring_meta else None
        if isinstance(score_meta, dict):
            bins = score_meta.get("bins") or [i / 10 for i in range(11)]
            reference = score_meta.get("reference") or [1.0 / (len(bins) - 1)] * (len(bins) - 1)
            self.score_monitor = DriftFeatureMonitor(
                name="score",
                bins=bins,
                reference=reference,
                window=cfg.window,
                emit_interval=cfg.emit_interval,
                psi_alert=cfg.psi_alert,
                ks_alert=cfg.ks_alert,
            )
        feature_meta = (monitoring_meta or {}).get("features") if monitoring_meta else None
        feature_meta = feature_meta or {}
        for feature_cfg in cfg.features:
            bins = feature_meta.get(feature_cfg.name, {}).get("bins") or feature_cfg.bins
            if not bins:
                continue
            reference = feature_meta.get(feature_cfg.name, {}).get("reference")
            if reference is None:
                reference = [1.0 / (len(bins) - 1)] * (len(bins) - 1)
            monitor = DriftFeatureMonitor(
                name=feature_cfg.name,
                bins=bins,
                reference=reference,
                window=cfg.window,
                emit_interval=cfg.emit_interval,
                psi_alert=feature_cfg.psi_alert or cfg.psi_alert,
                ks_alert=feature_cfg.ks_alert or cfg.ks_alert,
            )
            self.feature_monitors[feature_cfg.name] = monitor

    def update(self, score: float, features: Dict[str, float]) -> list[tuple[str, float, Dict[str, str]]]:
        if not self.enabled:
            return []
        metrics: list[tuple[str, float, Dict[str, str]]] = []
        if self.score_monitor is not None:
            metrics.extend(self._update_monitor(self.score_monitor, score, {"feature": "score"}))
        for name, monitor in self.feature_monitors.items():
            value = features.get(name)
            if value is None:
                continue
            metrics.extend(self._update_monitor(monitor, float(value), {"feature": name}))
        return metrics

    def _update_monitor(self, monitor: DriftFeatureMonitor, value: float, labels: Dict[str, str]) -> list[tuple[str, float, Dict[str, str]]]:
        try:
            monitor.update(float(value))
        except (TypeError, ValueError):
            return []
        if not monitor.ready():
            return []
        psi, ks = monitor.compute()
        metrics = [
            ("feature_psi", psi, labels),
            ("feature_ks", ks, labels),
        ]
        if psi > monitor.psi_alert:
            alert_labels = dict(labels)
            alert_labels["kind"] = "psi"
            metrics.append(("drift_alert_total", 1.0, alert_labels))
        if ks > monitor.ks_alert:
            alert_labels = dict(labels)
            alert_labels["kind"] = "ks"
            metrics.append(("drift_alert_total", 1.0, alert_labels))
        return metrics


class CalibrationMonitor:
    def __init__(self, bins, window: int, brier_alert: float, ece_alert: float) -> None:
        self.bins = np.array(bins, dtype=float)
        if self.bins.ndim != 1 or self.bins.size < 2:
            self.bins = np.linspace(0.0, 1.0, 11)
        self.window = max(window, 100)
        self.records: Deque[tuple[float, float]] = deque(maxlen=self.window)
        self.brier_alert = brier_alert
        self.ece_alert = ece_alert
        self.counter = 0

    def update(self, prob: float, outcome: float) -> list[tuple[str, float, Dict[str, str]]]:
        self.records.append((float(prob), float(outcome)))
        self.counter += 1
        if len(self.records) < min(self.window, 100) or self.counter % 25 != 0:
            return []
        probs = np.array([p for p, _ in self.records])
        outcomes = np.array([o for _, o in self.records])
        brier = float(np.mean((probs - outcomes) ** 2))
        hist, edges = np.histogram(probs, bins=self.bins)
        total = float(hist.sum()) or 1.0
        ece = 0.0
        for idx, count in enumerate(hist):
            if count == 0:
                continue
            lower = edges[idx]
            upper = edges[idx + 1]
            if idx == hist.size - 1:
                mask = (probs >= lower) & (probs <= upper)
            else:
                mask = (probs >= lower) & (probs < upper)
            if not np.any(mask):
                continue
            bucket_probs = probs[mask]
            bucket_outcomes = outcomes[mask]
            mean_conf = float(bucket_probs.mean())
            mean_outcome = float(bucket_outcomes.mean())
            ece += (count / total) * abs(mean_outcome - mean_conf)
        metrics = [
            ("calibration_brier_score", brier, {}),
            ("calibration_expected_calibration_error", float(ece), {}),
        ]
        if brier > self.brier_alert:
            metrics.append(("drift_alert_total", 1.0, {"kind": "brier"}))
        if ece > self.ece_alert:
            metrics.append(("drift_alert_total", 1.0, {"kind": "ece"}))
        return metrics


class ShadowMetrics:
    def __init__(self, window: int, threshold: float, promote_delta: float) -> None:
        self.window = max(window, 100)
        self.threshold = threshold
        self.promote_delta = promote_delta
        self.records: Deque[tuple[float, float, float]] = deque(maxlen=self.window)
        self.counter = 0

    def record(self, active_score: float, shadow_score: float | None, outcome: float) -> list[tuple[str, float, Dict[str, str]]]:
        if shadow_score is None:
            return []
        label = 1.0 if outcome >= 0.0 else 0.0
        self.records.append((float(active_score), float(shadow_score), label))
        self.counter += 1
        if len(self.records) < min(self.window, 120) or self.counter % 50 != 0:
            return []
        arr = np.array(self.records)
        active_scores = arr[:, 0]
        shadow_scores = arr[:, 1]
        labels = arr[:, 2]
        metrics: list[tuple[str, float, Dict[str, str]]] = []
        if labels.sum() > 0:
            ap_active = float(average_precision_score(labels, active_scores))
            ap_shadow = float(average_precision_score(labels, shadow_scores))
        else:
            ap_active = ap_shadow = 0.0
        metrics.append(("shadow_average_precision", ap_active, {"model": "active"}))
        metrics.append(("shadow_average_precision", ap_shadow, {"model": "shadow"}))
        metrics.append(("shadow_average_precision_delta", ap_shadow - ap_active, {}))
        thresh = self.threshold
        metrics.extend(self._precision_metric("active", active_scores, labels, thresh))
        metrics.extend(self._precision_metric("shadow", shadow_scores, labels, thresh))
        if ap_shadow - ap_active > self.promote_delta:
            metrics.append(("shadow_outperform_total", 1.0, {"delta": f"{(ap_shadow - ap_active):.4f}"}))
        return metrics

    def _precision_metric(self, model: str, scores: np.ndarray, labels: np.ndarray, threshold: float) -> list[tuple[str, float, Dict[str, str]]]:
        mask = scores >= threshold
        if not np.any(mask):
            precision = 0.0
        else:
            precision = float(labels[mask].mean())
        return [("shadow_precision_at_threshold", precision, {"model": model, "threshold": f"{threshold:.2f}"})]


class ProbabilityCalibrator:
    def __init__(self, w: Optional[float], b: Optional[float]) -> None:
        self.w = w
        self.b = b

    def apply(self, p: float) -> float:
        if self.w is None or self.b is None:
            return float(p)
        z = self.w * float(p) + self.b
        try:
            return 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            return 1.0 if z > 0 else 0.0

    @classmethod
    def from_meta(cls, meta: Dict[str, Any] | None, enabled: bool) -> "ProbabilityCalibrator":
        if not enabled or not meta:
            return cls(None, None)
        w = meta.get("w")
        b = meta.get("b")
        if w is None or b is None:
            return cls(None, None)
        return cls(float(w), float(b))


@dataclass
@dataclass
class ModelHandle:
    role: str
    model_id: str
    feature_names: List[str]
    session: Optional[object]
    input_name: Optional[str]
    prob_idx: Optional[int]
    calibrator: "ProbabilityCalibrator"
    thresholds: Dict[str, float]
    metadata: Dict[str, Any]
    path: Path


class AdaptiveThresholdManager:
    def __init__(self, cfg, thresholds, rs: RedisStream) -> None:
        self.cfg = cfg
        self.thresholds = thresholds
        self.rs = rs
        self.results: Deque[DecisionRecord] = deque(maxlen=max(cfg.window, 1) * 2)
        self.precision_history: Deque[float] = deque(maxlen=max(cfg.safety_windows, 1))
        self.last_adjustment_ts: float = 0.0
        self.frozen = False
        self.freeze_reason: Optional[str] = None
        self.base_entry = cfg.base_entry if cfg.base_entry is not None else thresholds.entry

    def record_result(self, record: DecisionRecord) -> None:
        if record.pnl is None:
            return
        self.results.append(record)

    def register_precision(self, value: float) -> None:
        self.precision_history.append(value)
        if len(self.precision_history) == self.precision_history.maxlen:
            if all(p < self.cfg.safety_precision for p in self.precision_history):
                if not self.frozen:
                    self._freeze(reason="precision_drop")
            elif self.frozen and any(p >= self.cfg.freeze_reset_precision for p in self.precision_history):
                self._unfreeze()

    def _freeze(self, reason: str) -> None:
        self.frozen = True
        self.freeze_reason = reason
        new_entry = min(self.thresholds.entry + self.cfg.step, self.cfg.max_entry)
        if new_entry != self.thresholds.entry:
            self.thresholds.entry = min(new_entry, self.base_entry + self.cfg.max_delta)
            log.warning("tau_entry_frozen", reason=reason, tau_entry=self.thresholds.entry)
            self._emit_tau_metrics()
        self.last_adjustment_ts = time.monotonic()

    def _unfreeze(self) -> None:
        if self.frozen:
            log.info("tau_entry_unfrozen", reason=self.freeze_reason)
        self.frozen = False
        self.freeze_reason = None

    def maybe_adjust(self) -> None:
        if not self.cfg.enabled:
            return
        now = time.monotonic()
        if now - self.last_adjustment_ts < self.cfg.interval_sec:
            return
        if len(self.results) < max(self.cfg.window, 1):
            return

        window_results = list(self.results)[-self.cfg.window :]
        candidates = [self.thresholds.entry]
        step = self.cfg.step
        candidates.extend([self.thresholds.entry - step, self.thresholds.entry + step])
        scored: List[Tuple[float, float, float]] = []
        min_count = max(5, int(len(window_results) * 0.2))
        for tau in candidates:
            tau = max(self.cfg.min_entry, min(self.cfg.max_entry, tau))
            selected = [r for r in window_results if r.score >= tau and r.pnl is not None]
            if len(selected) < min_count:
                continue
            successes = sum(1 for r in selected if r.pnl >= 0)
            precision = successes / len(selected) if selected else 0.0
            avg_pnl = sum(r.pnl for r in selected if r.pnl is not None) / len(selected)
            if avg_pnl < self.cfg.min_avg_pnl:
                continue
            scored.append((tau, precision, avg_pnl))

        if not scored:
            self.last_adjustment_ts = now
            return

        scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
        best_tau, best_precision, best_pnl = scored[0]
        current = self.thresholds.entry
        if best_tau == current:
            self.last_adjustment_ts = now
            return
        if best_tau < current and self.frozen:
            self.last_adjustment_ts = now
            return
        if abs(best_tau - self.base_entry) > self.cfg.max_delta:
            best_tau = self.base_entry + math.copysign(self.cfg.max_delta, best_tau - self.base_entry)
        best_tau = max(self.cfg.min_entry, min(self.cfg.max_entry, best_tau))
        if best_tau != current:
            direction = "raise" if best_tau > current else "lower"
            self.thresholds.entry = best_tau
            self.last_adjustment_ts = now
            log.info(
                "tau_entry_adjusted",
                direction=direction,
                tau_entry=best_tau,
                precision=best_precision,
                avg_pnl=best_pnl,
                frozen=self.frozen,
            )
            self._emit_tau_metrics()

    def _emit_tau_metrics(self) -> None:
        labels = {"split": "online", "model": "active"}
        self.rs.xadd("metrics:ai", Metric(name="tau_entry_current", value=float(self.thresholds.entry), labels=labels))
        self.rs.xadd("metrics:ai", Metric(name="tau_hold_current", value=float(self.thresholds.hold), labels=labels))


def load_metadata(path: str) -> Dict[str, Any] | None:
    try:
        meta_path = Path(path)
        if not meta_path.exists():
            log.warning("metadata_missing", path=str(meta_path))
            return None
        return json.loads(meta_path.read_text())
    except Exception as exc:
        log.warning("metadata_load_failed", path=path, error=str(exc))
        return None


def try_load_onnx(path: str) -> Tuple[Optional[object], Optional[str], Optional[int]]:
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        outputs = sess.get_outputs()
        prob_idx = 0
        for idx, out in enumerate(outputs):
            name = out.name.lower()
            if "prob" in name or "probability" in name:
                prob_idx = idx
                break
        return sess, input_name, prob_idx
    except Exception as exc:
        log.warning("onnx_load_failed", path=path, error=str(exc))
        return None, None, None


def extract_probability(outputs: List[np.ndarray], prob_idx: int) -> float:
    if not outputs:
        return 0.0
    prob_tensor = outputs[prob_idx]
    if isinstance(prob_tensor, np.ndarray):
        if prob_tensor.ndim == 2:
            if prob_tensor.shape[1] == 1:
                return float(prob_tensor[0, 0])
            if prob_tensor.shape[1] >= 2:
                return float(prob_tensor[0, -1])
        if prob_tensor.ndim == 1:
            return float(prob_tensor[-1])
    for out in outputs:
        if isinstance(out, dict):
            for key, value in out.items():
                try:
                    klass = str(key)
                    if klass.endswith("1") or klass in {"1", "positive", "True"}:
                        return float(value)
                except Exception:
                    continue
        if isinstance(out, np.ndarray) and out.size:
            return float(out.flat[-1])
    return 0.0


def build_feature_vector(cand: Candidate, feature_names: List[str]) -> np.ndarray:
    vec = np.zeros(len(feature_names), dtype=np.float32)
    feats = cand.features or {}
    for idx, name in enumerate(feature_names):
        vec[idx] = float(feats.get(name, 0.0))
    return vec.reshape(1, -1)


def _validate_feature_schema(metadata: Dict[str, Any] | None, state: FeatureState, feature_names: List[str]) -> None:
    if not metadata:
        raise SystemExit("Missing metadata for feature schema validation")
    schema_meta = metadata.get("feature_schema") or {}
    expected_version = getattr(state, "version", None)
    meta_version = schema_meta.get("version")
    if expected_version and meta_version and str(expected_version) != str(meta_version):
        raise SystemExit(f"Feature schema version mismatch: runtime={expected_version}, meta={meta_version}")
    meta_names = schema_meta.get("names")
    if meta_names:
        if feature_names and list(feature_names) != list(meta_names):
            raise SystemExit("Feature schema order mismatch")
    schema_hash = schema_meta.get("hash")
    runtime_hash = getattr(state, "config_hash", None)
    if schema_hash and runtime_hash and schema_hash != runtime_hash:
        raise SystemExit("Feature config hash mismatch")


class AIScorerService:
    def __init__(self) -> None:
        setup_logging()
        self.cfg = load_app_config()
        self.rs = RedisStream(self.cfg.redis.url, default_maxlen=self.cfg.redis.streams_maxlen)
        feature_state = FeatureState.from_config(self.cfg.signal_engine.features)
        default_thresholds = {
            "tau_entry": float(self.cfg.ai_scorer.thresholds.entry),
            "tau_hold": float(self.cfg.ai_scorer.thresholds.hold),
        }
        active_handle = load_registry_model(
            "active",
            feature_state,
            calibration_enabled=self.cfg.ai_scorer.calibration.enabled,
            default_thresholds=default_thresholds,
            allow_schema_mismatch=False,
        )
        self.metadata = active_handle.metadata
        self.feature_names = list(active_handle.feature_names) or _get_feature_order(feature_state)
        if not self.feature_names:
            self.feature_names = _get_feature_order(feature_state)
        self.pending: Dict[str, Deque[DecisionRecord]] = defaultdict(deque)
        self.recent_scores = RunningStats()
        self.score_metric_interval = 50
        self.score_metric_counter = 0
        monitoring_meta = (self.metadata or {}).get("monitoring")
        self.drift_monitor = DriftMonitor(self.cfg.ai_scorer.drift, monitoring_meta)
        drift_cfg = self.cfg.ai_scorer.drift
        if drift_cfg and getattr(drift_cfg, "enabled", False):
            bins = (monitoring_meta or {}).get("score", {}).get("bins") or drift_cfg.score_bins or [i / 10 for i in range(11)]
            self.calibration_monitor = CalibrationMonitor(bins, drift_cfg.window, drift_cfg.brier_alert, drift_cfg.ece_alert)
        else:
            self.calibration_monitor = None
        shadow_cfg = self.cfg.ai_scorer.shadow
        shadow_handle = None
        if shadow_cfg.enabled:
            shadow_handle = load_registry_model(
                "shadow",
                feature_state,
                calibration_enabled=self.cfg.ai_scorer.calibration.enabled,
                default_thresholds=default_thresholds,
                allow_schema_mismatch=True,
                shadow_cfg=shadow_cfg,
            )
            if shadow_handle is None:
                log.info("shadow_disabled", reason="load_failed")
        else:
            log.info("shadow_disabled", reason="config_disabled")
        log.info(
            "model_loaded",
            role="active",
            model_id=active_handle.model_id,
            schema_hash=active_handle.metadata.get("schema_hash"),
            path=str(active_handle.path),
        )
        if shadow_handle:
            log.info(
                "model_loaded",
                role="shadow",
                model_id=shadow_handle.model_id,
                schema_hash=shadow_handle.metadata.get("schema_hash"),
                path=str(shadow_handle.path),
            )
        self.models: Dict[str, ModelHandle] = {"active": active_handle}
        if shadow_handle:
            self.models["shadow"] = shadow_handle
        self.active_model_name = "active"
        self.last_model_switch = time.monotonic()
        self.deployment_cfg = self.cfg.ai_scorer.deployment
        self.active_handle = active_handle
        self.shadow_handle = shadow_handle
        self.cfg.ai_scorer.thresholds.entry = float(active_handle.thresholds.get("tau_entry", self.cfg.ai_scorer.thresholds.entry))
        self.cfg.ai_scorer.thresholds.hold = float(active_handle.thresholds.get("tau_hold", self.cfg.ai_scorer.thresholds.hold))
        self._emit_model_identifiers()
        self._emit_tau_metric("active", active_handle.thresholds)
        if self.shadow_handle:
            self._emit_tau_metric("shadow", self.shadow_handle.thresholds)
        self.shadow_metrics = None

        self.adaptive = AdaptiveThresholdManager(self.cfg.ai_scorer.adaptive, self.cfg.ai_scorer.thresholds, self.rs)
        self.window_buffer = None
        if self.cfg.ai_scorer.model_type.startswith("seq"):
            win_cfg = self.cfg.ai_scorer.window
            self.window_buffer = WindowBuffer(win_cfg.seconds, win_cfg.frequency_hz, win_cfg.depth_levels)
        self._emit_model_metrics()
        self.adaptive._emit_tau_metrics()

    def _activate_model(self, name: str, *, reason: str | None = None) -> None:
        handle = self.models.get(name)
        if handle is None:
            return
        self.active_model_name = name
        self.active_handle = handle
        self.cfg.ai_scorer.thresholds.entry = float(handle.thresholds.get("tau_entry", self.cfg.ai_scorer.thresholds.entry))
        self.cfg.ai_scorer.thresholds.hold = float(handle.thresholds.get("tau_hold", self.cfg.ai_scorer.thresholds.hold))
        self.last_model_switch = time.monotonic()
        log.info("model_switch", active=name, reason=reason)
        self._emit_model_identifiers()
        self._emit_tau_metric("active", handle.thresholds)
        if self.shadow_handle:
            self._emit_tau_metric("shadow", self.shadow_handle.thresholds)

    def _trigger_fallback(self, reason: str) -> None:
        if "shadow" in self.models and self.models.get("shadow") is not None:
            log.warning("model_fallback", reason=reason)
            self._activate_model("shadow", reason=reason)
        else:
            log.warning("model_fallback_unavailable", reason=reason)

    def _maybe_restore_model(self) -> None:
        if self.active_model_name == "active":
            return
        cooldown = max(int(getattr(self.cfg.ai_scorer.deployment, "cooldown_sec", 0)), 0)
        if cooldown <= 0:
            return
        if time.monotonic() - self.last_model_switch >= cooldown and "active" in self.models:
            log.info("model_restore", target="active")
            self._activate_model("active", reason="cooldown_expired")

    def _emit_model_identifiers(self) -> None:
        active = getattr(self, "active_handle", None)
        if active:
            self.rs.xadd(
                "metrics:ai",
                Metric(name="model_active_id", value=1.0, labels={"id": str(active.model_id)}),
            )
        else:
            self.rs.xadd(
                "metrics:ai",
                Metric(name="model_active_id", value=0.0, labels={"id": "none"}),
            )
        if getattr(self, "shadow_handle", None):
            self.rs.xadd(
                "metrics:ai",
                Metric(name="model_shadow_id", value=1.0, labels={"id": str(self.shadow_handle.model_id)}),
            )
        else:
            self.rs.xadd(
                "metrics:ai",
                Metric(name="model_shadow_id", value=0.0, labels={"id": "none"}),
            )

    def _emit_tau_metric(self, role: str, thresholds: Dict[str, float]) -> None:
        self.rs.xadd(
            "metrics:ai",
            Metric(
                name="tau_entry_current",
                value=float(thresholds.get("tau_entry", 0.0)),
                labels={"split": "online", "model": role},
            ),
        )
        self.rs.xadd(
            "metrics:ai",
            Metric(
                name="tau_hold_current",
                value=float(thresholds.get("tau_hold", 0.0)),
                labels={"split": "online", "model": role},
            ),
        )

    def run(self) -> None:
        streams = ["sig:candidates", "exec:fills", "metrics:risk", "control:events"]
        if self.window_buffer is not None:
            streams.append("md:raw")
        group = "aiscore"
        consumer = "c1"
        max_iters_env = int(os.environ.get("DRY_RUN_MAX_ITER", "0")) if "DRY_RUN_MAX_ITER" in os.environ else None
        processed = 0
        while True:
            self._maybe_restore_model()
            msgs = self.rs.read_group(group=group, consumer=consumer, streams=streams, block_ms=1000, count=self.cfg.ai_scorer.batch_size)
            if not msgs:
                self.adaptive.maybe_adjust()
                continue
            for stream, items in msgs:
                for msg_id, data in items:
                    try:
                        if stream == "md:raw" and self.window_buffer is not None:
                            from ..common.schema import MarketEvent

                            ev = MarketEvent.model_validate(data)
                            self.window_buffer.push(ev)
                            continue
                        if stream == "exec:fills":
                            fill = Fill.model_validate(data)
                            self._handle_fill(fill)
                        elif stream == "metrics:risk":
                            metric = Metric.model_validate(data)
                            self._handle_risk_metric(metric)
                        elif stream == "control:events":
                            evt = ControlEvent.model_validate(data)
                            self._handle_control_event(evt)
                        else:
                            cand = Candidate.model_validate(data)
                            self._score_candidate(cand)
                            processed += 1
                    except Exception as exc:
                        log.error("ai_scorer_error", stream=stream, error=str(exc))
                    finally:
                        self.rs.ack(stream, group, msg_id)
                    if max_iters_env and processed >= max_iters_env:
                        log.info("ai_scorer_exit_dry_run", processed=processed)
                        return
            self.adaptive.maybe_adjust()

    def _handle_risk_metric(self, metric: Metric) -> None:
        if metric.name == "rolling_precision":
            self.adaptive.register_precision(metric.value)
            if (
                self.deployment_cfg.auto_failover
                and metric.value < self.deployment_cfg.rollback_precision
            ):
                self._trigger_fallback("precision_floor")

    def _handle_control_event(self, evt: ControlEvent) -> None:
        if evt.type.upper() == "ROLLBACK":
            self._trigger_fallback(evt.reason or "control_event")
        elif evt.type.upper() == "MODEL_USE":
            target = (evt.details or {}).get("model") if evt.details else None
            if isinstance(target, str) and target in self.models:
                self._activate_model(target, reason=evt.reason or "manual_switch")
        elif evt.type.upper() == "MODEL_ACTIVE":
            if "active" in self.models:
                self._activate_model("active", reason=evt.reason or "manual_active")

    def _apply_spread_entry(self, base_entry: float, feats: Dict[str, float]) -> float:
        cfg = self.cfg.ai_scorer.spread_adaptive
        if not cfg.enabled:
            return float(base_entry)
        adj = 0.0
        spread_regime = int(feats.get("spread_regime", 1))
        if cfg.spread_regime_adjustments:
            idx = min(max(spread_regime, 0), len(cfg.spread_regime_adjustments) - 1)
            adj += cfg.spread_regime_adjustments[idx]
        vol_regime = int(feats.get("volatility_regime", 1))
        if cfg.volatility_regime_penalty:
            idx = min(max(vol_regime, 0), len(cfg.volatility_regime_penalty) - 1)
            adj += cfg.volatility_regime_penalty[idx]
        clamp_min, clamp_max = cfg.clamp
        dynamic = float(base_entry + adj)
        if clamp_min is not None:
            dynamic = max(clamp_min, dynamic)
        if clamp_max is not None:
            dynamic = min(clamp_max, dynamic)
        return dynamic

    def _score_candidate(self, cand: Candidate) -> None:
        feats = cand.features or {}
        decision_id = gen_id("dec_")
        feats.setdefault("decision_id", decision_id)
        scene_id = feats.get("scene_id")
        if scene_id is None:
            scene_id = f"{cand.symbol}-{int(feats.get('spread_regime', 0))}-{int(feats.get('volatility_regime', 0))}"
            feats["scene_id"] = scene_id
        self.rs.xadd("metrics:ai", Metric(name="signals_evaluated_total", value=1.0, labels={"symbol": cand.symbol}))
        active_handle = self.models[self.active_model_name]
        vector_active = build_feature_vector(cand, active_handle.feature_names)
        default_score = float(cand.score)
        p_raw = self._predict(active_handle, vector_active, default_score)
        p_cal = active_handle.calibrator.apply(p_raw)
        self.recent_scores.update(p_cal)
        self.score_metric_counter += 1
        if self.score_metric_counter >= self.score_metric_interval:
            self._emit_score_metrics()
            self.score_metric_counter = 0
        drift_metrics = self.drift_monitor.update(p_cal, feats)
        if drift_metrics:
            self._emit_metrics(drift_metrics)
        shadow_cal: Optional[float] = None
        if self.shadow_handle is not None:
            shadow_handle = self.shadow_handle
            shadow_vector = build_feature_vector(cand, shadow_handle.feature_names)
            shadow_raw = self._predict(shadow_handle, shadow_vector, default_score)
            shadow_cal = shadow_handle.calibrator.apply(shadow_raw)
            feats["shadow_p_calibrated"] = float(shadow_cal)
            self.rs.xadd(
                "metrics:ai",
                Metric(name="shadow_signals_evaluated_total", value=1.0, labels={"symbol": cand.symbol}),
            )
            self.rs.xadd(
                "metrics:ai",
                Metric(name="shadow_score_distribution", value=float(shadow_cal), labels={"split": "online"}),
            )
            shadow_tau_entry = self._apply_spread_entry(shadow_handle.thresholds.get("tau_entry", 0.0), feats)
            if shadow_cal >= shadow_tau_entry:
                self.rs.xadd(
                    "metrics:ai",
                    Metric(name="shadow_signals_approved_total", value=1.0, labels={"symbol": cand.symbol}),
                )
        base_tau_entry = self.cfg.ai_scorer.thresholds.entry
        dynamic_tau_entry = self._apply_spread_entry(base_tau_entry, feats)
        feats.update({
            "p_raw": float(p_raw),
            "p_calibrated": float(p_cal),
            "tau_entry_base": float(base_tau_entry),
            "tau_entry": float(dynamic_tau_entry),
            "tau_hold": float(self.cfg.ai_scorer.thresholds.hold),
        })
        adjustment = dynamic_tau_entry - base_tau_entry
        if abs(adjustment) > 1e-6:
            feats["tau_entry_adjustment"] = float(adjustment)
            self.rs.xadd(
                "metrics:ai",
                Metric(name="tau_entry_spread_adjust", value=float(adjustment), labels={"symbol": cand.symbol}),
            )
        if p_cal >= dynamic_tau_entry:
            record = DecisionRecord(
                decision_id=decision_id,
                symbol=cand.symbol,
                score=float(p_cal),
                ts_ms=utc_ms(),
                shadow_score=float(shadow_cal) if shadow_cal is not None else None,
            )
            self.pending[cand.symbol].append(record)
            appr = ApprovedSignal(**cand.model_dump(), p_success=float(p_cal))
            self.rs.xadd("sig:approved", appr)
            self.rs.xadd("metrics:ai", Metric(name="signals_approved_total", value=1.0, labels={"symbol": cand.symbol}))

    def _handle_fill(self, fill: Fill) -> None:
        queue = self.pending.get(fill.symbol)
        if not queue:
            return
        record = queue.popleft()
        record.pnl = fill.pnl
        self.adaptive.record_result(record)
        metrics: list[tuple[str, float, Dict[str, str]]] = []
        if self.calibration_monitor is not None:
            metrics.extend(self.calibration_monitor.update(record.score, 1.0 if (fill.pnl or 0.0) >= 0.0 else 0.0))
        if self.shadow_metrics is not None:
            metrics.extend(self.shadow_metrics.record(record.score, record.shadow_score, float(fill.pnl or 0.0)))
        if metrics:
            self._emit_metrics(metrics)

    def _emit_score_metrics(self) -> None:
        mean = self.recent_scores.mean
        std = self.recent_scores.std
        labels = {"split": "online"}
        self.rs.xadd(
            "metrics:ai", Metric(name="model_score_distribution", value=mean, labels={"stat": "mean", **labels})
        )
        self.rs.xadd(
            "metrics:ai", Metric(name="model_score_distribution", value=std, labels={"stat": "std", **labels})
        )


def main():
    service = AIScorerService()
    service.run()


if __name__ == "__main__":
    main()
