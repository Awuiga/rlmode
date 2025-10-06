from __future__ import annotations

import random
import time
from typing import Dict

from .base import ExchangeAdapter
from ..common.schema import Side, TIF, Order
from ..common.utils import gen_id


class FakeExchange(ExchangeAdapter):
    def __init__(self, symbols: Dict[str, Dict[str, float]]):
        self._symbols = symbols
        self._orders: Dict[str, Order] = {}

    @property
    def name(self) -> str:
        return "fake"

    def place_limit_post_only(self, *, symbol: str, side: Side, price: float, qty: float, tif: TIF) -> Order:
        cid = gen_id("ord_")
        order = Order(
            client_id=cid,
            symbol=symbol,
            side=side,
            price=price,
            qty=qty,
            tif=tif,
            post_only=True,
            state="NEW",
        )
        self._orders[cid] = order
        # Simulate small delay
        time.sleep(0.005)
        return order

    def cancel(self, *, symbol: str, client_id: str) -> bool:
        return self._orders.pop(client_id, None) is not None

    def get_min_steps(self, *, symbol: str) -> Dict[str, float]:
        meta = self._symbols.get(symbol)
        if not meta:
            return {"tick_size": 0.01, "lot_step": 0.001, "min_qty": 0.001}
        return {
            "tick_size": float(meta.get("tick_size", 0.01)),
            "lot_step": float(meta.get("lot_step", 0.001)),
            "min_qty": float(meta.get("min_qty", 0.001)),
        }

