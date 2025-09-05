from __future__ import annotations

import random
from typing import List

from ..common.config import load_app_config
from ..common.logging import setup_logging, get_logger
from ..common.redis_stream import RedisStream
from ..common.schema import Candidate, ApprovedSignal, Metric
from ..common.utils import utc_ms
from .window_buffer import WindowBuffer

log = get_logger("ai_scorer")


def try_load_onnx(path: str):
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return sess
    except Exception as e:
        log.warning("onnx_load_failed", error=str(e))
        return None


def main():
    setup_logging()
    cfg = load_app_config()
    rs = RedisStream(cfg.redis.url)

    sess = try_load_onnx(cfg.ai_scorer.model_path)
    win_cfg = cfg.ai_scorer.window
    wbuf = WindowBuffer(win_cfg.seconds, win_cfg.frequency_hz, win_cfg.depth_levels)

    log.info("ai_scorer_start")

    group = "aiscore"
    consumer = "c1"
    # Also subscribe to market data to maintain the L2/L1 window buffer
    streams = ["md:raw", "sig:candidates"]

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
                    if stream == "md:raw":
                        # Feed buffer with market data
                        from ..common.schema import MarketEvent

                        ev = MarketEvent.model_validate(data)
                        wbuf.push(ev)
                    else:
                        cand = Candidate.model_validate(data)
                        # Use window if available to influence scoring, otherwise fallback to cand.score
                        _ = wbuf.build_tensor(cand.symbol)
                        if sess is None:
                            p_succ = max(0.0, min(1.0, cand.score + random.uniform(-0.05, 0.05)))
                        else:
                            # Placeholder mapping to ONNX input. Real impl should construct ndarray.
                            p_succ = max(0.0, min(1.0, cand.score))

                        if p_succ >= cfg.ai_scorer.threshold:
                            appr = ApprovedSignal(**cand.model_dump(), p_success=p_succ)
                            rs.xadd("sig:approved", appr)
                            rs.xadd("metrics:ai", Metric(name="approved_total", value=1.0))
                            processed += 1
                except Exception as e:
                    log.error("ai_scorer_error", error=str(e))
                finally:
                    rs.ack(stream, group, msg_id)
                if max_iters_env and processed >= max_iters_env:
                    log.info("ai_scorer_exit_dry_run", processed=processed)
                    return


if __name__ == "__main__":
    main()
