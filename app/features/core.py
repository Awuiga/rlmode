from __future__ import annotations

import json
from typing import TYPE_CHECKING, Dict, Iterable, List, Tuple

from ..common.schema import LastTrade, MarketEvent, TradeSide
from ..signal_engine.features import FeatureState as FeatureState  # noqa: F401 re-export
from ..signal_engine.features import compute_features as compute_features  # noqa: F401 re-export

if TYPE_CHECKING:  # pragma: no cover
    from ..common.config import FeatureConfig

__all__ = ["FeatureState", "compute_features", "compute_features_offline", "make_feature_state"]


def make_feature_state(feature_cfg: FeatureConfig) -> FeatureState:  # type: ignore[name-defined]
    """Helper to construct runtime state from the shared feature config."""
    return FeatureState.from_config(feature_cfg)


def _as_depth(raw_levels: object) -> List[Tuple[float, float]]:
    if raw_levels is None:
        return []
    levels = raw_levels
    if isinstance(levels, str):
        try:
            levels = json.loads(levels)
        except json.JSONDecodeError:
            return []
    result: List[Tuple[float, float]] = []
    if not isinstance(levels, Iterable):
        return result
    for level in levels:
        try:
            price = float(level[0])
            qty = float(level[1])
            result.append((price, qty))
        except (TypeError, ValueError, IndexError):
            continue
    return result


def _as_last_trade(raw_trade: object) -> LastTrade | None:
    if raw_trade is None:
        return None
    trade = raw_trade
    if isinstance(trade, str):
        try:
            trade = json.loads(trade)
        except json.JSONDecodeError:
            return None
    if not isinstance(trade, dict):
        return None
    try:
        price = float(trade.get("price"))
        qty = float(trade.get("qty"))
        side_raw = trade.get("side")
        if isinstance(side_raw, str):
            side = TradeSide(side_raw.lower())
        elif isinstance(side_raw, TradeSide):
            side = side_raw
        else:
            return None
        return LastTrade(price=price, qty=qty, side=side)
    except (TypeError, ValueError, KeyError):
        return None


def compute_features_offline(row: Dict[str, object], state: FeatureState, depth_top_k: int) -> Dict[str, float]:
    """Offline wrapper using the online feature state to avoid drift."""
    ts_val = row.get("ts") or row.get("ts_ms")
    try:
        ts = int(float(ts_val)) if ts_val is not None else 0
    except (TypeError, ValueError):
        ts = 0
    symbol = str(row.get("symbol") or row.get("sym") or "")
    try:
        bid1 = float(row.get("bid1") or row.get("bid_px"))
    except (TypeError, ValueError):
        bid1 = 0.0
    try:
        ask1 = float(row.get("ask1") or row.get("ask_px"))
    except (TypeError, ValueError):
        ask1 = 0.0

    bids = _as_depth(row.get("bids") or row.get("b"))
    asks = _as_depth(row.get("asks") or row.get("a"))
    last_trade = _as_last_trade(row.get("last_trade"))

    ev = MarketEvent(
        ts=ts,
        symbol=symbol,
        bid1=bid1,
        ask1=ask1,
        bids=bids,
        asks=asks,
        last_trade=last_trade,
    )
    return state.update(ev, depth_top_k=depth_top_k)
