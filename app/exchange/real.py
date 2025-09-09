from __future__ import annotations

from typing import Dict

from .base import ExchangeAdapter
from .binance_futs import BinanceFuturesAdapter
from ..common.schema import Side, TIF, Order


class RealExchange(ExchangeAdapter):
    def __init__(self, *, use: str):
        self._use = use
        if use == "binance_futs":
            self._impl = BinanceFuturesAdapter()
        else:
            # Placeholder for Bybit v5 implementation
            raise NotImplementedError("Bybit v5 real adapter is not yet implemented")

    @property
    def name(self) -> str:
        return f"real:{self._use}"

    def place_limit_post_only(self, *, symbol: str, side: Side, price: float, qty: float, tif: TIF) -> Order:
        return self._impl.place_limit_post_only(symbol=symbol, side=side, price=price, qty=qty, tif=tif)

    def cancel(self, *, symbol: str, client_id: str) -> bool:
        return self._impl.cancel(symbol=symbol, client_id=client_id)

    def get_min_steps(self, *, symbol: str) -> Dict[str, float]:
        return self._impl.get_min_steps(symbol=symbol)

