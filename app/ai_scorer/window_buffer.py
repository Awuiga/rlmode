from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Tuple

from ..common.schema import MarketEvent


class WindowBuffer:
    def __init__(self, seconds: int, frequency_hz: int, depth_levels: int):
        self.max_len = seconds * frequency_hz + 5
        self.depth_levels = depth_levels
        self.buffers: Dict[str, Deque[MarketEvent]] = {}

    def push(self, ev: MarketEvent):
        buf = self.buffers.setdefault(ev.symbol, deque(maxlen=self.max_len))
        buf.append(ev)

    def ready(self, symbol: str) -> bool:
        return len(self.buffers.get(symbol, [])) >= int(self.max_len * 0.8)

    def build_tensor(self, symbol: str):
        # Placeholder: return simple features array instead of real L2 tensor
        # In production, build [T, F] with L2 snapshots at fixed time grid
        buf = list(self.buffers.get(symbol, []))
        if not buf:
            return None
        # Example derived features: [mid, spread, topk_bid_qty, topk_ask_qty]
        feats = []
        for ev in buf[-self.max_len :]:
            mid = (ev.bid1 + ev.ask1) / 2.0
            spread = (ev.ask1 - ev.bid1)
            bq = sum(q for _, q in ev.bids[: self.depth_levels])
            aq = sum(q for _, q in ev.asks[: self.depth_levels])
            feats.append([mid, spread, bq, aq])
        return feats

