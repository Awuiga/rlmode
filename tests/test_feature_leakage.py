from __future__ import annotations

from typing import List

import pytest

pytestmark = pytest.mark.unit

from app.common.schema import LastTrade, MarketEvent, TradeSide
from app.features.core import FeatureState, compute_features, compute_features_offline


def _make_event(
    ts: int,
    bid: float,
    ask: float,
    bids: List[List[float]] | List[tuple[float, float]],
    asks: List[List[float]] | List[tuple[float, float]],
    trade_side: TradeSide | None = None,
) -> MarketEvent:
    levels_bid = [(float(p), float(q)) for p, q in bids]
    levels_ask = [(float(p), float(q)) for p, q in asks]
    last = None
    if trade_side is not None:
        last = LastTrade(price=(bid + ask) / 2.0, qty=0.05, side=trade_side)
    return MarketEvent(
        ts=ts,
        symbol='BTCUSDT',
        bid1=float(bid),
        ask1=float(ask),
        bids=levels_bid,
        asks=levels_ask,
        last_trade=last,
    )


def _sample_events() -> List[MarketEvent]:
    return [
        _make_event(1_000, 100.0, 100.2, [(100.0, 1.2), (99.9, 1.0)], [(100.2, 0.9), (100.3, 1.1)]),
        _make_event(1_050, 100.1, 100.25, [(100.1, 1.0), (100.0, 1.4)], [(100.25, 1.2), (100.35, 0.8)], TradeSide.buy),
        _make_event(1_120, 100.05, 100.2, [(100.05, 1.5), (99.95, 1.2)], [(100.2, 0.8), (100.3, 0.7)], TradeSide.sell),
        _make_event(1_180, 100.08, 100.24, [(100.08, 1.3), (99.98, 1.1)], [(100.24, 0.85), (100.34, 0.9)], TradeSide.buy),
        _make_event(1_250, 100.12, 100.28, [(100.12, 1.4), (100.02, 1.2)], [(100.28, 0.95), (100.38, 0.8)], TradeSide.sell),
    ]


def test_online_features_do_not_use_future_information():
    events = _sample_events()
    field_order = FeatureState(1500, 1000).field_order

    state_all = FeatureState(1500, 1000)
    features_all = []
    for ev in events:
        features_all.append(compute_features(ev, state_all, depth_top_k=2))

    for idx in range(len(events)):
        prefix_state = FeatureState(1500, 1000)
        last_feat = None
        for j in range(idx + 1):
            last_feat = compute_features(events[j], prefix_state, depth_top_k=2)
        assert last_feat is not None
        for key in field_order:
            assert last_feat[key] == pytest.approx(features_all[idx][key], rel=1e-6, abs=1e-8)


def _row_from_event(ev: MarketEvent) -> dict:
    return {
        "ts": ev.ts,
        "symbol": ev.symbol,
        "bid1": ev.bid1,
        "ask1": ev.ask1,
        "bids": [(p, q) for p, q in ev.bids],
        "asks": [(p, q) for p, q in ev.asks],
        "last_trade": ev.last_trade.model_dump() if ev.last_trade else None,
    }


def test_offline_features_do_not_use_future_information():
    events = _sample_events()
    rows = [_row_from_event(ev) for ev in events]
    field_order = FeatureState(1500, 1000).field_order

    state_all = FeatureState(1500, 1000)
    features_all = []
    for row in rows:
        features_all.append(compute_features_offline(row, state_all, depth_top_k=2))

    for idx in range(len(rows)):
        prefix_state = FeatureState(1500, 1000)
        last_feat = None
        for j in range(idx + 1):
            last_feat = compute_features_offline(rows[j], prefix_state, depth_top_k=2)
        assert last_feat is not None
        for key in field_order:
            assert last_feat[key] == pytest.approx(features_all[idx][key], rel=1e-6, abs=1e-8)
