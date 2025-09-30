from __future__ import annotations

from typing import Dict

from .base import ExchangeAdapter
from .binance_futs import BinanceFuturesAdapter
from .bybit_perp import BybitPerpAdapter
from ..common.schema import Side, TIF, Order


class RealExchange(ExchangeAdapter):
    def __init__(self, *, venue: str, maker_only: bool = True, reduce_only: bool = True, post_only: bool = True):
        venue_normalized = venue.lower()
        self._venue = venue_normalized
        self._maker_only = maker_only
        self._reduce_only = reduce_only
        self._post_only = post_only
        if venue_normalized in {"binance_futs", "binance"}:
            self._impl = BinanceFuturesAdapter()
        elif venue_normalized in {"bybit_v5", "bybit"}:
            self._impl = BybitPerpAdapter()
        else:
            raise ValueError(f"Unsupported exchange venue: {venue}")

    @property
    def name(self) -> str:
        return f"real:{self._venue}"

    def place_limit_post_only(self, *, symbol: str, side: Side, price: float, qty: float, tif: TIF) -> Order:
        order = self._impl.place_limit_post_only(symbol=symbol, side=side, price=price, qty=qty, tif=tif)
        if self._post_only and not getattr(order, "post_only", True):
            # Enforce maker-only behaviour without chasing the book
            self._impl.cancel(symbol=symbol, client_id=order.client_id)
            raise RuntimeError("post-only enforcement triggered; order would take liquidity")
        order.post_only = True
        return order

    def cancel(self, *, symbol: str, client_id: str) -> bool:
        return self._impl.cancel(symbol=symbol, client_id=client_id)

    def get_min_steps(self, *, symbol: str) -> Dict[str, float]:
        return self._impl.get_min_steps(symbol=symbol)
