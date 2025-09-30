from __future__ import annotations

import asyncio
import json
import os
import random
import time
from typing import Dict, List, Optional

import websockets
import yaml

from ..common.config import load_app_config
from ..common.logging import setup_logging, get_logger
from ..common.redis_stream import RedisStream
from ..common.schema import MarketEvent, LastTrade, Metric
from ..common.utils import utc_ms


log = get_logger("collector")


def load_symbols_meta(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def fake_md_generator(symbol: str, tick_size: float, depth_levels: int):
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


async def binance_stream(rs: RedisStream, symbols: List[str], depth_levels: int, url_base: str):
    streams = []
    for s in symbols:
        sym = s.lower()
        streams.append(f"{sym}@depth20@100ms")
        streams.append(f"{sym}@aggTrade")
    url = f"{url_base}?streams={'/'.join(streams)}"
    last_trade: Dict[str, LastTrade] = {}
    backoff = [0.5, 1.0, 2.0, 5.0]
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                log.info("binance_ws_connected", url=url)
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=60)
                    msg = json.loads(raw)
                    stream = msg.get("stream", "")
                    data = msg.get("data", {})
                    if stream.endswith("aggtrade") or (data.get("e") == "aggTrade"):
                        sym = stream.split("@")[0].upper()
                        price = float(data.get("p", 0.0))
                        qty = float(data.get("q", 0.0))
                        side = "sell" if data.get("m", False) else "buy"
                        last_trade[sym] = LastTrade(price=price, qty=qty, side=side)
                        continue
                    # Depth event
                    bids_raw = data.get("b", [])
                    asks_raw = data.get("a", [])
                    if not bids_raw or not asks_raw:
                        continue
                    bids = [(float(p), float(q)) for p, q in bids_raw[:depth_levels]]
                    asks = [(float(p), float(q)) for p, q in asks_raw[:depth_levels]]
                    bid1 = float(bids[0][0])
                    ask1 = float(asks[0][0])
                    sym = stream.split("@")[0].upper()
                    evt = MarketEvent(
                        ts=int(data.get("E") or utc_ms()),
                        symbol=sym,
                        bid1=bid1,
                        ask1=ask1,
                        bids=bids,
                        asks=asks,
                        last_trade=last_trade.get(sym),
                    )
                    rs.xadd("md:raw", evt)
        except Exception as e:
            log.error("binance_ws_error", error=str(e))
            await asyncio.sleep(backoff[0])
            backoff = backoff[1:] + [backoff[-1]]


async def bybit_stream(rs: RedisStream, symbols: List[str], depth_levels: int, url_base: str, category: str):
    # Build primary and AWS fallback URLs as path-based category endpoints
    cat = (category or "linear").strip().lower()
    base = url_base.rstrip("/")
    primary_url = f"{base}/{cat}"
    aws_url = primary_url.replace("stream.bybit.com", "stream.bybit-aws.com")

    # Subscribe to standard Bybit v5 public topics
    args = []
    for s in symbols:
        args.append(f"publicTrade.{s}")
        args.append(f"orderbook.1.{s}")
        args.append(f"tickers.{s}")
    sub = json.dumps({"op": "subscribe", "args": args})

    last_trade: Dict[str, LastTrade] = {}
    orderbook: Dict[str, Dict[str, List[List[str]]]] = {}
    backoff = [0.5, 1.0, 2.0, 5.0]

    while True:
        # Try primary, then AWS fallback within the same iteration
        for url in (primary_url, aws_url):
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    log.info("bybit_ws_connected", url=url, symbols=symbols)
                    await ws.send(sub)
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=60)
                        msg = json.loads(raw)
                        topic = msg.get("topic") or ""
                        if not topic:
                            continue
                        if topic.startswith("publicTrade"):
                            data = msg.get("data", [])
                            for t in data:
                                sym = t.get("s") or t.get("symbol") or ""
                                if not sym:
                                    continue
                                price = float(t.get("p"))
                                qty = float(t.get("v") or t.get("q", 0.0))
                                side = "buy" if (t.get("S") or t.get("side")) == "Buy" else "sell"
                                last_trade[sym] = LastTrade(price=price, qty=qty, side=side)
                            continue
                        if topic.startswith("tickers"):
                            data = (msg.get("data") or [{}])[0] if isinstance(msg.get("data"), list) else (msg.get("data") or {})
                            sym = data.get("symbol") or data.get("s")
                            if sym:
                                # Derive side from tickDirection when available
                                dirv = data.get("tickDirection") or data.get("td") or ""
                                side = "buy" if "Plus" in str(dirv) else ("sell" if dirv else None)
                                try:
                                    px = float(data.get("lastPrice") or data.get("lp"))
                                except Exception:
                                    px = None
                                if px is not None and side is not None:
                                    last_trade[sym] = LastTrade(price=px, qty=0.0, side=side)  # qty unknown from ticker
                            continue
                        if topic.startswith("orderbook"):
                            sym = topic.split(".")[-1]
                            typ = msg.get("type")
                            data = msg.get("data") or {}
                            if typ == "snapshot":
                                orderbook[sym] = {"b": data.get("b", []), "a": data.get("a", [])}
                            elif typ == "delta":
                                book = orderbook.setdefault(sym, {"b": [], "a": []})
                                # Apply deltas (simplified): prepend new levels
                                if "b" in data:
                                    book["b"] = data["b"] + book.get("b", [])
                                if "a" in data:
                                    book["a"] = data["a"] + book.get("a", [])
                            book = orderbook.get(sym) or {}
                            bids_raw = book.get("b", [])[:depth_levels]
                            asks_raw = book.get("a", [])[:depth_levels]

                            # Robust float parsing with per-level try/except
                            def _levels(raw):
                                out = []
                                for lvl in raw or []:
                                    try:
                                        p = float(lvl[0])
                                        q = float(lvl[1])
                                        out.append((p, q))
                                    except Exception:
                                        # skip malformed level
                                        continue
                                return out

                            bids = _levels(bids_raw)
                            asks = _levels(asks_raw)
                            if not bids or not asks:
                                continue
                            bid1 = bids[0][0]
                            ask1 = asks[0][0]
                            evt = MarketEvent(
                                ts=int(msg.get("ts") or utc_ms()),
                                symbol=sym,
                                bid1=bid1,
                                ask1=ask1,
                                bids=bids,
                                asks=asks,
                                last_trade=last_trade.get(sym),
                            )
                            rs.xadd("md:raw", evt)
            except Exception as e:
                log.error("bybit_ws_error", error=str(e), url=url)
                # Try next URL in sequence (fallback) or backoff if both failed
                continue
            # If we exit the with-context without exception, break retry loop
            break
        else:
            # Both endpoints failed
            await asyncio.sleep(backoff[0])
            backoff = backoff[1:] + [backoff[-1]]


def main():
    setup_logging()
    cfg = load_app_config()
    rs = RedisStream(cfg.redis.url)

    symbols_meta = load_symbols_meta("config/symbols.yml")
    depth_levels = cfg.collector.depth_levels

    log.info("collector_start", exchange_mode=cfg.exchange_mode, market=cfg.market.use, symbols=cfg.symbols)

    max_iters_env = int(os.environ.get("DRY_RUN_MAX_ITER", "0")) if "DRY_RUN_MAX_ITER" in os.environ else None

    if cfg.market.use == "fake":
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
            rs.xadd("metrics:collector", Metric(name="collector_heartbeat", value=1.0), idempotency_key=None)
    else:
        # Live public WS (read-only)
        async def run_live():
            tasks = []
            if cfg.market.use == "binance_futs":
                tasks.append(asyncio.create_task(binance_stream(rs, cfg.symbols, depth_levels, cfg.market.binance_ws_public)))
            elif cfg.market.use == "bybit_v5":
                tasks.append(
                    asyncio.create_task(
                        bybit_stream(
                            rs,
                            cfg.symbols,
                            depth_levels,
                            cfg.market.bybit_ws_public,
                            getattr(cfg.market, "category", "linear"),
                        )
                    )
                )
            else:
                log.error("unsupported_market", market=cfg.market.use)
                return
            # Heartbeat metrics
            async def heartbeat():
                while True:
                    rs.xadd("metrics:collector", Metric(name="collector_heartbeat", value=1.0))
                    await asyncio.sleep(5)
            tasks.append(asyncio.create_task(heartbeat()))
            if max_iters_env:
                # Allow exit in dry-run CI
                await asyncio.sleep(max(1, max_iters_env // 50))
                for t in tasks:
                    t.cancel()
            else:
                await asyncio.gather(*tasks)

        asyncio.run(run_live())


if __name__ == "__main__":
    main()
