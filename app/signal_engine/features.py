from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

from ..common.schema import MarketEvent


class FeatureState:
    def __init__(self, sigma_window_ms: int, ofi_window_ms: int):
        self.mid_prices: Deque[Tuple[int, float]] = deque(maxlen=5000)
        self.trades: Deque[Tuple[int, float]] = deque(maxlen=5000)  # signed qty (+buy, -sell)
        self.sigma_window_ms = sigma_window_ms
        self.ofi_window_ms = ofi_window_ms

    def push(self, ev: MarketEvent):
        mid = (ev.bid1 + ev.ask1) / 2.0
        self.mid_prices.append((ev.ts, mid))
        if ev.last_trade:
            signed = ev.last_trade.qty if ev.last_trade.side == ev.last_trade.side.buy else -ev.last_trade.qty
            self.trades.append((ev.ts, signed))
        # Drop old entries by time windows
        cutoff_sigma = ev.ts - self.sigma_window_ms
        while self.mid_prices and self.mid_prices[0][0] < cutoff_sigma:
            self.mid_prices.popleft()
        cutoff_ofi = ev.ts - self.ofi_window_ms
        while self.trades and self.trades[0][0] < cutoff_ofi:
            self.trades.popleft()


def compute_features(ev: MarketEvent, state: FeatureState, depth_top_k: int) -> Dict[str, float]:
    features: Dict[str, float] = {}
    # Spread
    spread = max(0.0, ev.ask1 - ev.bid1)
    features["spread"] = spread

    # Sigma short: std of mid returns over window
    mids = [m for _, m in state.mid_prices]
    sigma = 0.0
    if len(mids) >= 3:
        rets = []
        for i in range(1, len(mids)):
            if mids[i - 1] > 0:
                rets.append((mids[i] - mids[i - 1]) / mids[i - 1])
        if len(rets) >= 2:
            mu = sum(rets) / len(rets)
            var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
            sigma = var ** 0.5
    features["sigma"] = sigma

    # OFI approximate: sum signed qty in window
    ofi = sum(q for _, q in state.trades)
    features["ofi"] = ofi

    # QI: imbalance top-K
    topk_bids = sum(q for _, q in ev.bids[:depth_top_k])
    topk_asks = sum(q for _, q in ev.asks[:depth_top_k])
    denom = (topk_bids + topk_asks) or 1.0
    qi = (topk_bids - topk_asks) / denom
    features["qi"] = qi

    # Depth cum
    features["depth_bids"] = topk_bids
    features["depth_asks"] = topk_asks

    return features

