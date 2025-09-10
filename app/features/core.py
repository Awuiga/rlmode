from __future__ import annotations

from typing import Dict

# Re-export the online feature impl to ensure a single source for features
from ..signal_engine.features import FeatureState as FeatureState  # noqa: F401
from ..signal_engine.features import compute_features as compute_features  # noqa: F401


def compute_features_offline(row: Dict[str, float], state: FeatureState, depth_top_k: int) -> Dict[str, float]:
    """
    Offline wrapper for computing features from a dict-like row with keys compatible to MarketEvent
    (ts, symbol, bid1, ask1, bids, asks, last_trade).

    This delegates to the same compute_features and thus should match online behavior.
    """
    # Minimal shim object to satisfy the state.push/compute expectations
    class _Ev:
        def __init__(self, r: Dict[str, float]):
            self.ts = int(r["ts"]) if "ts" in r else int(r["ts_ms"])  # tolerate offline naming
            self.symbol = str(r.get("symbol"))
            self.bid1 = float(r.get("bid1"))
            self.ask1 = float(r.get("ask1"))
            self.bids = r.get("bids", [])
            self.asks = r.get("asks", [])
            self.last_trade = r.get("last_trade")

    ev = _Ev(row)
    state.push(ev)  # update buffers
    from ..signal_engine.features import compute_features as _cf

    return _cf(ev, state, depth_top_k=depth_top_k)

