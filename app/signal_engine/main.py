from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime, timezone, time as dtime
from typing import Deque, Dict, Optional, Tuple

import os

import numpy as np

from ..common.config import load_app_config
from ..common.logging import setup_logging, get_logger
from ..common.redis_stream import RedisStream
from ..common.schema import MarketEvent, Candidate, Metric, Side
from ..common.utils import utc_ms
from ..features.core import FeatureState, compute_features
from ..common.profiling import LatencyProfiler
from .rules import apply_rules
from .backpressure import BackpressureController

HISTORY_WINDOW_MS = 1000


log = get_logger("signal_engine")


def _parse_time(value: str) -> dtime:
    hour, minute = value.split(":", 1)
    return dtime(int(hour), int(minute))


def _time_in_window(current: dtime, start: dtime, end: dtime) -> bool:
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end


def _session_state(ts_ms: int, session_cfg) -> Tuple[float, Optional[str]]:
    if not getattr(session_cfg, "enabled", False):
        return 1.0, None

    if getattr(session_cfg, "news_windows", None):
        for win in session_cfg.news_windows:
            if ts_ms >= int(win.start_ts) and ts_ms <= int(win.end_ts):
                return 1.0, "news"

    tighten = getattr(session_cfg, "tighten_multiplier", 1.0) or 1.0
    if getattr(session_cfg, "open_windows", None):
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        current = dt.time()
        for win in session_cfg.open_windows:
            start = _parse_time(win.start)
            end = _parse_time(win.end)
            if _time_in_window(current, start, end):
                return tighten, None
        return tighten, "session"
    return tighten, None


def _check_anomaly(feats: Dict[str, float], gates_cfg) -> None:
    if not getattr(gates_cfg.anomaly, "enabled", True):
        return
    z_limit = getattr(gates_cfg.anomaly, "z_limit", 0.0) or 0.0
    score = float(feats.get("anomaly_score", 0.0))
    if z_limit and score > z_limit and getattr(gates_cfg.anomaly, "require_double_confirmation", True):
        feats["requires_ai_confirmation"] = 1.0


def _check_gates(
    symbol: str,
    ts: int,
    feats: Dict[str, float],
    state: FeatureState,
    gates_cfg,
    feature_history: Dict[str, Deque[Tuple[int, Dict[str, float]]]],
) -> Optional[str]:
    tighten_factor, session_reason = _session_state(ts, gates_cfg.session)
    if session_reason:
        return session_reason

    _check_anomaly(feats, gates_cfg)

    # Spread gate with optional tightening
    if getattr(gates_cfg.spread, "enabled", True):
        max_spread = float(getattr(gates_cfg.spread, "max_ticks", 0.0) or 0.0)
        if tighten_factor > 1.0 and max_spread > 0:
            max_spread = max_spread / tighten_factor
        if max_spread > 0 and feats.get("spread", 0.0) > max_spread:
            return "spread"

    # Volatility gate
    if getattr(gates_cfg.volatility, "enabled", True):
        sigma = feats.get("sigma", 0.0)
        sigma_hist = [val for _, val in getattr(state, "sigma_history", [])]
        if sigma_hist:
            q = float(np.quantile(sigma_hist, min(max(getattr(gates_cfg.volatility, "quantile", 0.9), 0.0), 0.999)))
            if tighten_factor > 1.0:
                q = q / tighten_factor
            if sigma > q:
                return "volatility"

    # Liquidity gate
    if getattr(gates_cfg.liquidity, "enabled", True):
        min_bid = float(getattr(gates_cfg.liquidity, "min_bid_qty", 0.0) or 0.0)
        min_ask = float(getattr(gates_cfg.liquidity, "min_ask_qty", 0.0) or 0.0)
        if tighten_factor > 1.0:
            min_bid *= tighten_factor
            min_ask *= tighten_factor
        if feats.get("depth_bids", 0.0) < min_bid or feats.get("depth_asks", 0.0) < min_ask:
            return "liquidity"

    # Cross-asset gate with debounce
    if getattr(gates_cfg.cross_asset, "enabled", False):
        leader = getattr(gates_cfg.cross_asset, "leader_symbol", "").upper()
        if leader and symbol.upper() != leader:
            history = feature_history.get(leader, deque())
            debounce_window = int(getattr(gates_cfg.cross_asset, "debounce_window_ms", 200) or 200)
            debounce_count = int(getattr(gates_cfg.cross_asset, "debounce_count", 2) or 2)
            recent = [entry for entry in history if ts - entry[0] <= debounce_window]
            if len(recent) < debounce_count:
                return "leadlag"
            feature_name = getattr(gates_cfg.cross_asset, "feature", "microprice_velocity")
            leader_vals = [feat.get(feature_name, 0.0) for _, feat in recent[-debounce_count:]]
            candidate_value = feats.get(feature_name, 0.0)
            if not leader_vals:
                return "leadlag"
            direction = getattr(gates_cfg.cross_asset, "direction", "same").lower()
            leader_sign = np.sign(np.mean(leader_vals))
            candidate_sign = np.sign(candidate_value)
            if direction == "same" and leader_sign * candidate_sign <= 0:
                return "leadlag"
            if direction == "opposite" and leader_sign * candidate_sign >= 0:
                return "leadlag"
    return None


def main():
    setup_logging()
    if os.environ.get("RL_MODE_TEST_ENTRYPOINT") == "1":
        log.info("entrypoint_test_skip", service="signal_engine")
        return
    cfg = load_app_config()
    rs = RedisStream(
        cfg.redis.url,
        default_maxlen=cfg.redis.streams_maxlen,
        pending_retry=cfg.redis.pending_retry,
    )

    feature_cfg = cfg.signal_engine.features
    feat_state: Dict[str, FeatureState] = defaultdict(lambda: FeatureState.from_config(feature_cfg))
    feature_history: Dict[str, Deque[Tuple[int, Dict[str, float]]]] = defaultdict(lambda: deque(maxlen=64))

    log.info("signal_engine_start")

    latency_profiler = LatencyProfiler(
        rs=rs,
        stream="metrics:signal",
        metric_name="signal_pipeline_latency_ms",
        labels={"stage": "feature_rules"},
        window=256,
        emit_every=64,
    )
    backpressure = BackpressureController(cfg.signal_engine.backpressure, rs)

    group = "sigeng"
    consumer = "c1"
    streams = ["md:raw"]
    max_iters_env = int(os.environ.get("DRY_RUN_MAX_ITER", "0")) if "DRY_RUN_MAX_ITER" in os.environ else None
    processed = 0
    while True:
        msgs = rs.read_group(group=group, consumer=consumer, streams=streams, block_ms=1000, count=100)
        if not msgs:
            continue
        for stream, items in msgs:
            for msg_id, data in items:
                try:
                    ev = MarketEvent.model_validate(data)
                    if backpressure.should_throttle(ev.ts):
                        rs.xadd(
                            "metrics:signal",
                            Metric(name="backpressure_total", value=1.0, labels={"symbol": ev.symbol}),
                        )
                        continue
                    with latency_profiler.track():
                        st = feat_state[ev.symbol]
                        feats = compute_features(ev, st, depth_top_k=feature_cfg.depth_top_k)
                        feats.setdefault("signal_id", f"{ev.symbol}:{ev.ts}")
                        symbol_upper = ev.symbol.upper()
                        history = feature_history[symbol_upper]
                        history.append((ev.ts, dict(feats)))
                        while history and ev.ts - history[0][0] > HISTORY_WINDOW_MS:
                            history.popleft()
                        reason = _check_gates(
                            symbol=ev.symbol,
                            ts=ev.ts,
                            feats=feats,
                            state=st,
                            gates_cfg=cfg.signal_engine.gates,
                            feature_history=feature_history,
                        )
                        if reason:
                            rs.xadd(
                                "metrics:signal",
                                Metric(
                                    name="gate_drop_reason_total",
                                    value=1.0,
                                    labels={"reason": reason, "symbol": ev.symbol},
                                ),
                            )
                            continue
                        cand = apply_rules(
                            ts=ev.ts,
                            symbol=ev.symbol,
                            feats=feats,
                            side_hint=None,
                            cfg_rules=cfg.signal_engine.rules,
                            tp_pct=cfg.signal_engine.tp_pct,
                            sl_pct=cfg.signal_engine.sl_pct,
                            weights=cfg.signal_engine.scoring.weights.model_dump(),
                            theta_emit=cfg.signal_engine.scoring.theta_emit,
                        )
                    if cand:
                        signal_id = str(cand.features.get("signal_id") or f"{cand.symbol}:{cand.ts}")
                        cand.features["signal_id"] = signal_id
                        rs.xadd("sig:candidates", cand, idempotency_key=signal_id)
                        rs.xadd("metrics:signal", Metric(name="candidates_emitted_total", value=1.0))
                        processed += 1
                except Exception as e:  # parsing/validation errors shouldn't kill loop
                    log.error("signal_engine_error", error=str(e))
                finally:
                    rs.ack(stream, group, msg_id)
                if max_iters_env and processed >= max_iters_env:
                    log.info("signal_engine_exit_dry_run", processed=processed)
                    return


if __name__ == "__main__":
    main()
