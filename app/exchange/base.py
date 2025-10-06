from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional

from ..common.schema import Side, TIF, Order


class ExchangeAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:  # e.g., binance_futs, bybit_perp, fake
        ...

    # Execution
    @abstractmethod
    def place_limit_post_only(
        self, *, symbol: str, side: Side, price: float, qty: float, tif: TIF
    ) -> Order:
        ...

    @abstractmethod
    def cancel(self, *, symbol: str, client_id: str) -> bool:
        ...

    @abstractmethod
    def get_min_steps(self, *, symbol: str) -> Dict[str, float]:
        # returns {"tick_size": float, "lot_step": float, "min_qty": float}
        ...

