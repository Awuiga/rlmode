from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

from app.features.core import FeatureState, compute_features, compute_features_offline
from app.common.schema import MarketEvent, LastTrade


def _mk_ev(ts: int, bid: float, ask: float, bids, asks, last_side: str | None = None):
    lt = None
    if last_side:
        lt = LastTrade(price=(bid + ask) / 2.0, qty=0.01, side=last_side)
    return MarketEvent(ts=ts, symbol="BTCUSDT", bid1=bid, ask1=ask, bids=bids, asks=asks, last_trade=lt)


def test_features_online_offline_match():
    state_on = FeatureState(1500, 1000)
    state_off = FeatureState(1500, 1000)
    rows: list[Dict] = []

    # Build synthetic sequence
    events = [
        _mk_ev(1_000, 100.0, 101.0, [(100.0, 1.0)], [(101.0, 1.2)], None),
        _mk_ev(1_020, 100.1, 101.1, [(100.1, 0.9)], [(101.1, 1.1)], "buy"),
        _mk_ev(1_040, 100.2, 101.2, [(100.2, 0.8)], [(101.2, 1.0)], "sell"),
        _mk_ev(1_060, 100.1, 101.1, [(100.1, 1.1)], [(101.1, 1.0)], "buy"),
    ]

    for ev in events:
        feats_on = compute_features(ev, state_on, depth_top_k=1)
        # offline row proxy
        row = {
            "ts": ev.ts,
            "symbol": ev.symbol,
            "bid1": ev.bid1,
            "ask1": ev.ask1,
            "bids": ev.bids,
            "asks": ev.asks,
            "last_trade": ev.last_trade.model_dump() if ev.last_trade else None,
        }
        feats_off = compute_features_offline(row, state_off, depth_top_k=1)

        for k in ("spread", "mid", "sigma", "ofi", "qi", "bid1", "ask1"):
            assert pytest.approx(feats_on[k], rel=1e-6, abs=1e-9) == feats_off[k]
