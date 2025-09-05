from __future__ import annotations

from collections import deque
from typing import Deque

from ..common.config import load_app_config
from ..common.logging import setup_logging, get_logger
from ..common.redis_stream import RedisStream
from ..common.schema import Fill, Metric, ControlEvent
from ..common.utils import utc_ms
import yaml


log = get_logger("risk")


def main():
    setup_logging()
    cfg = load_app_config()
    rs = RedisStream(cfg.redis.url)

    with open("config/risk.yml", "r", encoding="utf-8") as f:
        risk_cfg = yaml.safe_load(f) or {}

    daily_pnl = 0.0
    consecutive_losses = 0
    window: Deque[float] = deque(maxlen=int(risk_cfg.get("fill_rate_window", 50)))
    fill_rate_min = float(risk_cfg.get("fill_rate_min", 0.25))
    daily_loss_limit = float(risk_cfg.get("daily_loss_limit", -0.03))
    markout_max = float(risk_cfg.get("markout_max", 0.01))

    log.info("risk_start")

    group = "risk"
    consumer = "c1"
    streams = ["exec:fills"]

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
                    fill = Fill.model_validate(data)
                    # Use pnl field from fill; for fake it's 0, so approximate by markout
                    trade_pnl = fill.pnl
                    daily_pnl += trade_pnl
                    win = 1.0 if trade_pnl > 0 else 0.0
                    consecutive_losses = 0 if trade_pnl > 0 else (consecutive_losses + 1)
                    window.append(win)
                    fill_rate = sum(window) / (len(window) or 1)

                    rs.xadd("metrics:risk", Metric(name="daily_pnl", value=daily_pnl))
                    rs.xadd("metrics:risk", Metric(name="winrate", value=fill_rate))

                    trigger = None
                    reason = ""
                    if daily_pnl <= daily_loss_limit:
                        trigger = "STOP"
                        reason = f"daily_pnl<=limit ({daily_pnl:.4f}<= {daily_loss_limit})"
                    elif consecutive_losses >= int(risk_cfg.get("max_consecutive_losses", 5)):
                        trigger = "STOP"
                        reason = f"consecutive_losses >= N ({consecutive_losses})"
                    elif len(window) >= window.maxlen and fill_rate < fill_rate_min:
                        trigger = "STOP"
                        reason = f"fill_rate below min ({fill_rate:.2%}<{fill_rate_min:.2%})"

                    if trigger:
                        evt = ControlEvent(ts=utc_ms(), type=trigger, reason=reason)
                        rs.xadd("control:events", evt)
                        rs.xadd("metrics:risk", Metric(name="kill_switch", value=1.0))
                        # Do not exit process; just continue monitoring
                except Exception as e:
                    log.error("risk_error", error=str(e))
                finally:
                    rs.ack(stream, group, msg_id)
                processed += 1
                if max_iters_env and processed >= max_iters_env:
                    log.info("risk_exit_dry_run", processed=processed)
                    return


if __name__ == "__main__":
    main()
