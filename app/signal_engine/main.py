from __future__ import annotations

from collections import defaultdict
from typing import Dict

from ..common.config import load_app_config
from ..common.logging import setup_logging, get_logger
from ..common.redis_stream import RedisStream
from ..common.schema import MarketEvent, Candidate, Metric, Side
from ..common.utils import utc_ms
from ..features.core import FeatureState, compute_features
from .rules import apply_rules


log = get_logger("signal_engine")


def main():
    setup_logging()
    cfg = load_app_config()
    rs = RedisStream(cfg.redis.url)

    feat_state: Dict[str, FeatureState] = defaultdict(lambda: FeatureState(cfg.signal_engine.features.sigma_window_ms, cfg.signal_engine.features.ofi_window_ms))

    log.info("signal_engine_start")

    group = "sigeng"
    consumer = "c1"
    streams = ["md:raw"]
    import os
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
                    st = feat_state[ev.symbol]
                    st.push(ev)
                    feats = compute_features(ev, st, depth_top_k=cfg.signal_engine.features.depth_top_k)
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
                        rs.xadd("sig:candidates", cand)
                        rs.xadd("metrics:signal", Metric(name="candidates_total", value=1.0))
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
