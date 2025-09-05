from __future__ import annotations

# NOTE: Placeholder adapter skeleton. Production implementation should handle:
# - REST HMAC signing (timestamp, recvWindow)
# - Post-only via GTX
# - User data WS for fills
# - Rate limits, retries, backoff

import hmac
import os
import time
from hashlib import sha256
from typing import Dict, Optional

import httpx

from .base import ExchangeAdapter
from ..common.schema import Side, TIF, Order


class BinanceFuturesAdapter(ExchangeAdapter):
    def __init__(self, *, api_key: Optional[str] = None, api_secret: Optional[str] = None, base_url: str = "https://fapi.binance.com"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
        self.client = httpx.Client(base_url=self.base_url, timeout=5.0)
        if not (self.api_key and self.api_secret):
            # Allow instantiation for dry-run but real calls will fail
            pass
    @property
    def name(self) -> str:
        return "binance_futs"

    def _sign(self, params: Dict[str, str]) -> Dict[str, str]:
        if not self.api_secret:
            raise RuntimeError("BINANCE_API_SECRET is required for signed endpoints")
        query = "&".join([f"{k}={v}" for k, v in params.items()])
        sig = hmac.new(self.api_secret.encode(), query.encode(), sha256).hexdigest()
        return {**params, "signature": sig}

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise RuntimeError("BINANCE_API_KEY is required for signed endpoints")
        return {"X-MBX-APIKEY": self.api_key}

    def place_limit_post_only(self, *, symbol: str, side: Side, price: float, qty: float, tif: TIF) -> Order:
        # timeInForce GTX indicates post-only on futures
        now = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "side": side.value,
            "type": "LIMIT",
            "timeInForce": "GTX",
            "price": f"{price}",
            "quantity": f"{qty}",
            "timestamp": str(now),
            "recvWindow": "2500",
            "newClientOrderId": f"ord_{now}",
        }
        signed = self._sign(params)
        r = self.client.post("/fapi/v1/order", headers=self._headers(), params=signed)
        r.raise_for_status()
        data = r.json()
        oid = str(data.get("clientOrderId") or data.get("orderId"))
        return Order(
            client_id=oid,
            exchange_id=str(data.get("orderId")),
            symbol=symbol,
            side=side,
            price=price,
            qty=qty,
            tif=tif,
            post_only=True,
            state=str(data.get("status", "NEW")),
        )

    def cancel(self, *, symbol: str, client_id: str) -> bool:
        now = int(time.time() * 1000)
        params = {"symbol": symbol, "origClientOrderId": client_id, "timestamp": str(now), "recvWindow": "2500"}
        signed = self._sign(params)
        r = self.client.delete("/fapi/v1/order", headers=self._headers(), params=signed)
        if r.status_code == 200:
            return True
        # Not found or already canceled
        if r.status_code == 400 and "Unknown order" in r.text:
            return False
        r.raise_for_status()
        return True

    def get_min_steps(self, *, symbol: str) -> Dict[str, float]:
        # For live trading, fetch exchange info filters; by default, leave to external config
        r = self.client.get("/fapi/v1/exchangeInfo")
        if r.status_code != 200:
            # Fallback defaults
            return {"tick_size": 0.01, "lot_step": 0.001, "min_qty": 0.001}
        data = r.json()
        for s in data.get("symbols", []):
            if s.get("symbol") == symbol:
                tick_size = 0.01
                lot_step = 0.001
                min_qty = 0.001
                for f in s.get("filters", []):
                    if f.get("filterType") == "PRICE_FILTER":
                        tick_size = float(f.get("tickSize", tick_size))
                    if f.get("filterType") == "LOT_SIZE":
                        lot_step = float(f.get("stepSize", lot_step))
                        min_qty = float(f.get("minQty", min_qty))
                return {"tick_size": tick_size, "lot_step": lot_step, "min_qty": min_qty}
        return {"tick_size": 0.01, "lot_step": 0.001, "min_qty": 0.001}
