from __future__ import annotations

import random
import time
from typing import Dict

import yaml

from ..common.config import load_app_config
from ..common.logging import setup_logging, get_logger
from ..common.redis_stream import RedisStream
from ..common.schema import ApprovedSignal, Order, Fill, TIF, Side, Metric
from ..common.utils import utc_ms
from .ladder import build_ladder
from .positions import Position
from ..exchange.fake import FakeExchange


log = get_logger("executor")


def load_symbols_meta(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    setup_logging()
    cfg = load_app_config()
    rs = RedisStream(cfg.redis.url)
    symbols_meta = load_symbols_meta("config/symbols.yml")

    # For now, only fake exchange is implemented end-to-end
    ex = FakeExchange(symbols_meta)
    pos = {sym: Position() for sym in cfg.symbols}

    log.info("executor_start", exchange=ex.name)

    group = "exec"
    consumer = "c1"
    streams = ["sig:approved"]

    import os
    max_iters_env = int(os.environ.get("DRY_RUN_MAX_ITER", "0")) if "DRY_RUN_MAX_ITER" in os.environ else None
    processed = 0
    while True:
        msgs = rs.read_group(group=group, consumer=consumer, streams=streams, block_ms=1000, count=50)
        if not msgs:
            continue
        for stream, items in msgs:
            for msg_id, data in items:
                try:
                    sig = ApprovedSignal.model_validate(data)
                    sym = sig.symbol
                    meta = symbols_meta.get(sym, {})
                    tick = float(meta.get("tick_size", 0.01))
                    lot = float(meta.get("lot_step", 0.001))
                    min_qty = float(meta.get("min_qty", 0.001))

                    mid = (sig.features.get("mid") or (sig.features.get("spread", 0) + 0.0))
                    # If mid not present, approximate from spread and ask1/bid1 not available; fallback constant
                    if mid == 0:
                        mid = 30000.0

                    qty = 0.01  # Small nominal size for demo
                    ladder = build_ladder(
                        side=sig.side,
                        mid_price=mid,
                        qty=qty,
                        fractions=cfg.executor.ladder.fractions,
                        step_ticks=cfg.executor.ladder.step_ticks,
                        tick_size=tick,
                        lot_step=lot,
                        min_qty=min_qty,
                        use_percent=cfg.executor.ladder.use_percent,
                    )
                    for price, q in ladder:
                        order = ex.place_limit_post_only(symbol=sym, side=sig.side, price=price, qty=q, tif=TIF[cfg.executor.tif])
                        # Simulate maker fills based on simple odds
                        fill_prob = 0.6 if sig.side == Side.BUY else 0.6
                        if random.random() < fill_prob:
                            fill = Fill(
                                order_id=order.client_id,
                                symbol=sym,
                                side=sig.side,
                                price=price,
                                qty=q,
                                fee=0.0,
                                pnl=(-0.01 if os.environ.get("DRY_RUN_FORCE_LOSSES") == "1" else random.choice([-0.002, 0.002])),
                                markout=0.0,
                                is_maker=True,
                                ts=utc_ms(),
                            )
                            # Update position simple PnL baseline (markout remains 0 for demo)
                            pos[sym].apply_fill(sig.side.value, price, q)
                            rs.xadd("exec:fills", fill)
                            rs.xadd("metrics:executor", Metric(name="fills_total", value=1.0))
                        else:
                            # Not filled; simulate cancel on timeout
                            time.sleep(cfg.executor.cancel_timeout_ms / 1000.0)
                            ex.cancel(symbol=sym, client_id=order.client_id)
                except Exception as e:
                    log.error("executor_error", error=str(e))
                finally:
                    rs.ack(stream, group, msg_id)
                processed += 1
                if max_iters_env and processed >= max_iters_env:
                    log.info("executor_exit_dry_run", processed=processed)
                    return


if __name__ == "__main__":
    main()
