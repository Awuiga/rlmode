from __future__ import annotations

import random
import time
from typing import Dict, Set

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
    halted = False
    open_orders: Set[str] = set()

    log.info("executor_start", exchange=ex.name)

    group = "exec"
    consumer = "c1"
    streams = ["sig:approved", "control:events"]

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
                    if stream == "control:events":
                        # Stop new orders; cancel anything open
                        halted = True
                        for oid in list(open_orders):
                            try:
                                # We don't know symbol mapping here; fake exchange only tracks by id
                                ex.cancel(symbol=cfg.symbols[0], client_id=oid)
                            except Exception:
                                pass
                            finally:
                                open_orders.discard(oid)
                        rs.xadd("metrics:executor", Metric(name="open_orders", value=float(len(open_orders))))
                        log.warning("executor_halted")
                    else:
                        if halted:
                            # Do not place new orders, just ack and continue
                            continue
                        sig = ApprovedSignal.model_validate(data)
                        sym = sig.symbol
                        meta = symbols_meta.get(sym, {})
                        tick = float(meta.get("tick_size", 0.01))
                        lot = float(meta.get("lot_step", 0.001))
                        min_qty = float(meta.get("min_qty", 0.001))

                        # Determine anchor price from top of book
                        bid1 = sig.features.get("bid1", 0.0)
                        ask1 = sig.features.get("ask1", 0.0)
                        anchor = bid1 if sig.side == Side.BUY else ask1
                        if anchor <= 0:
                            anchor = sig.features.get("mid", 0.0) or 30000.0

                        # Apply entry_ref offsets (ticks or percent)
                        entry_price = anchor
                        if sig.entry_ref.offset_ticks is not None:
                            if sig.side == Side.BUY:
                                entry_price = anchor - sig.entry_ref.offset_ticks * tick
                            else:
                                entry_price = anchor + sig.entry_ref.offset_ticks * tick
                        elif sig.entry_ref.offset_pct is not None:
                            if sig.side == Side.BUY:
                                entry_price = anchor * (1 - sig.entry_ref.offset_pct)
                            else:
                                entry_price = anchor * (1 + sig.entry_ref.offset_pct)
                        elif sig.entry_ref.price is not None:
                            entry_price = sig.entry_ref.price

                        qty = 0.01  # Small nominal size for demo
                        ladder = build_ladder(
                            side=sig.side,
                            mid_price=entry_price,
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
                            open_orders.add(order.client_id)
                            rs.xadd("metrics:executor", Metric(name="open_orders", value=float(len(open_orders))))
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
                                # Update position PnL baseline
                                pos[sym].apply_fill(sig.side.value, price, q)
                                rs.xadd("exec:fills", fill)
                                rs.xadd("metrics:executor", Metric(name="trades_total", value=1.0))
                                rs.xadd("metrics:executor", Metric(name="maker_fills_total", value=1.0))
                                open_orders.discard(order.client_id)
                                rs.xadd("metrics:executor", Metric(name="open_orders", value=float(len(open_orders))))
                            else:
                                # Not filled; simulate cancel on timeout
                                time.sleep(cfg.executor.cancel_timeout_ms / 1000.0)
                                ex.cancel(symbol=sym, client_id=order.client_id)
                                if order.client_id in open_orders:
                                    open_orders.discard(order.client_id)
                                    rs.xadd("metrics:executor", Metric(name="open_orders", value=float(len(open_orders))))
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
