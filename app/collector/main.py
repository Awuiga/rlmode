from __future__ import annotations

import random
import time
from typing import Dict
import os

from ..common.config import load_app_config
from ..common.logging import setup_logging, get_logger
from ..common.redis_stream import RedisStream
from ..common.schema import MarketEvent, LastTrade, Metric
from ..common.utils import utc_ms
import yaml


log = get_logger("collector")


def load_symbols_meta(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def fake_md_generator(symbol: str, tick_size: float, depth_levels: int):
    # Simple random walk mid price
    mid = 30000.0 if symbol.endswith("USDT") else 2000.0
    vol = 2.0
    last_side = "buy"
    while True:
        mid += random.gauss(0, vol)
        mid = max(10 * tick_size, mid)
        spread = max(tick_size, random.choice([tick_size, 2 * tick_size]))
        bid1 = mid - spread / 2
        ask1 = mid + spread / 2
        bids = []
        asks = []
        for i in range(depth_levels):
            p_bid = bid1 - i * tick_size
            p_ask = ask1 + i * tick_size
            bids.append((round(p_bid, 10), round(random.uniform(0.01, 0.5), 6)))
            asks.append((round(p_ask, 10), round(random.uniform(0.01, 0.5), 6)))
        if random.random() < 0.3:
            last_side = "buy" if last_side == "sell" else "sell"
        last_trade = LastTrade(price=round(mid, 10), qty=round(random.uniform(0.001, 0.01), 6), side=last_side)
        evt = MarketEvent(
            ts=utc_ms(), symbol=symbol, bid1=round(bid1, 10), ask1=round(ask1, 10), bids=bids, asks=asks, last_trade=last_trade
        )
        yield evt
        time.sleep(0.02)  # ~50 Hz


def main():
    setup_logging()
    cfg = load_app_config()
    rs = RedisStream(cfg.redis.url)

    symbols_meta = load_symbols_meta("config/symbols.yml")
    depth_levels = cfg.collector.depth_levels

    log.info("collector_start", exchange=cfg.exchange, symbols=cfg.symbols, depth_levels=depth_levels)

    max_iters_env = int(os.environ.get("DRY_RUN_MAX_ITER", "0")) if "DRY_RUN_MAX_ITER" in os.environ else None

    if cfg.exchange == "fake":
        gens = {}
        for sym in cfg.symbols:
            tick = float(symbols_meta.get(sym, {}).get("tick_size", 0.01))
            gens[sym] = fake_md_generator(sym, tick, depth_levels)
        processed = 0
        while True:
            for sym, gen in gens.items():
                evt = next(gen)
                rs.xadd("md:raw", evt, idempotency_key=None)
                processed += 1
                if max_iters_env and processed >= max_iters_env:
                    log.info("collector_exit_dry_run", processed=processed)
                    return
            # Lightweight heartbeat metric
            rs.xadd("metrics:collector", Metric(name="collector_heartbeat", value=1.0), idempotency_key=None)
    else:
        # Placeholder for real WS adapters; implement reconnect/backoff/subscribe/ping here.
        log.error("exchange_not_implemented", exchange=cfg.exchange)
        time.sleep(5)


if __name__ == "__main__":
    main()
