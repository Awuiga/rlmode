from __future__ import annotations

# NOTE: Placeholder adapter skeleton. Production implementation should handle:
# - REST/WS signing
# - Post-only flag
# - User data WS for fills
# - Rate limits, retries, backoff

from typing import Dict

from .base import ExchangeAdapter
from ..common.schema import Side, TIF, Order


class BybitPerpAdapter(ExchangeAdapter):
    @property
    def name(self) -> str:
        return "bybit_perp"

    def place_limit_post_only(self, *, symbol: str, side: Side, price: float, qty: float, tif: TIF) -> Order:
        raise NotImplementedError("Implement Bybit Perp REST order placement")

    def cancel(self, *, symbol: str, client_id: str) -> bool:
        raise NotImplementedError

    def get_min_steps(self, *, symbol: str) -> Dict[str, float]:
        # Should query instrument info or be provided via symbols.yml
        raise NotImplementedError

