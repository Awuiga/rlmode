"""
Legacy entry point kept for compatibility.
Updated to use Redis Streams and unified schemas.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from app.common.config import load_app_config
from app.common.logging import setup_logging
from app.common.redis_stream import RedisStream
from app.common.schema import Candidate, EntryRef, Side, MarketEvent

LOGGER = logging.getLogger("signals_legacy")


def _build_candidate_from_raw(event: Dict[str, Any], tp: float, sl: float) -> Candidate | None:
    # Fix: read bids/asks from separate fields
    bids = event.get("data", {}).get("b", [])
    asks = event.get("data", {}).get("a", [])
    if not bids or not asks:
        return None
    bid = float(bids[0][0])
    ask = float(asks[0][0])
    spread = ask - bid

    # Minimal rule: require positive spread
    if spread <= 0:
        return None

    symbol = event.get("stream", "").split("@")[0].upper() or event.get("s", "")
    ts = int(event.get("E") or event.get("ts") or 0)

    features = {"spread": float(spread), "bid1": bid, "ask1": ask, "mid": (bid + ask) / 2.0}
    side = Side.BUY if spread > 0 else Side.SELL
    entry_ref = EntryRef(type="limit", offset_ticks=1)
    return Candidate(ts=ts, symbol=symbol, side=side, entry_ref=entry_ref, tp_pct=tp, sl_pct=sl, features=features, score=0.5)


def main():
    setup_logging()
    cfg = load_app_config()
    rs = RedisStream(cfg.redis.url)
    LOGGER.info("legacy_signal_engine_start")
    group = "sigeng_legacy"
    consumer = "c1"
    streams = ["md:raw"]

    while True:
        msgs = rs.read_group(group=group, consumer=consumer, streams=streams, block_ms=1000, count=100)
        if not msgs:
            continue
        for stream, items in msgs:
            for msg_id, data in items:
                try:
                    # data might already be normalized MarketEvent; handle both
                    cand: Candidate | None
                    try:
                        ev = MarketEvent.model_validate(data)
                        bids = ev.bids
                        asks = ev.asks
                        if not bids or not asks:
                            cand = None
                        else:
                            features = {"spread": ev.ask1 - ev.bid1, "bid1": ev.bid1, "ask1": ev.ask1, "mid": (ev.ask1 + ev.bid1) / 2}
                            side = Side.BUY if features["spread"] > 0 else Side.SELL
                            cand = Candidate(ts=ev.ts, symbol=ev.symbol, side=side, entry_ref=EntryRef(type="limit", offset_ticks=1), tp_pct=cfg.signal_engine.tp_pct, sl_pct=cfg.signal_engine.sl_pct, features=features, score=0.5)
                    except Exception:
                        cand = _build_candidate_from_raw(data, cfg.signal_engine.tp_pct, cfg.signal_engine.sl_pct)
                    if cand:
                        rs.xadd("sig:candidates", cand)
                finally:
                    rs.ack(stream, group, msg_id)


if __name__ == "__main__":  # pragma: no cover
    main()
