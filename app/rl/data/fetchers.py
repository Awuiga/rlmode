"""Async data fetchers for historical spot, LOB, funding and on-chain metrics."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"


@dataclass(slots=True)
class TradeTick:
    timestamp: datetime
    price: float
    quantity: float
    is_buyer_maker: bool


@dataclass(slots=True)
class OrderBookLevel:
    price: float
    quantity: float


@dataclass(slots=True)
class OrderBookSnapshot:
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]

    @property
    def spread(self) -> float:
        if not self.bids or not self.asks:
            return math.inf
        return max(0.0, self.asks[0].price - self.bids[0].price)

    @property
    def mid_price(self) -> float:
        if not self.bids or not self.asks:
            levels = (self.bids + self.asks)
            return levels[0].price if levels else float("nan")
        return 0.5 * (self.bids[0].price + self.asks[0].price)

    def depth(self, side: str, top_k: int = 10) -> float:
        levels = self.bids if side == "bid" else self.asks
        return float(sum(level.quantity for level in levels[:top_k]))

    def imbalance(self, top_k: int = 5) -> float:
        bid_depth = self.depth("bid", top_k=top_k)
        ask_depth = self.depth("ask", top_k=top_k)
        denom = bid_depth + ask_depth
        if denom <= 0.0:
            return 0.0
        return float((bid_depth - ask_depth) / denom)


@dataclass(slots=True)
class FundingRate:
    timestamp: datetime
    rate: float
    symbol: str


@dataclass(slots=True)
class OnChainMetric:
    timestamp: datetime
    name: str
    value: float


class _RateLimiter:
    """Simple coroutine-friendly token bucket used to respect exchange limits."""

    def __init__(self, *, max_rate_per_sec: float, burst: int = 5) -> None:
        self.interval = 1.0 / max_rate_per_sec
        self.tokens = float(burst)
        self.capacity = float(burst)
        self.updated_at = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.updated_at
            self.updated_at = now
            self.tokens = min(self.capacity, self.tokens + elapsed / self.interval)
            if self.tokens < 1.0:
                await asyncio.sleep((1.0 - self.tokens) * self.interval)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0


class BinanceSpotFetcher:
    """Fetch spot trades and L2 order book snapshots from Binance REST endpoints."""

    def __init__(
        self,
        *,
        symbol: str,
        http_client: Optional[httpx.AsyncClient] = None,
        requests_per_second: float = 8.0,
        snapshot_limit: int = 1000,
    ) -> None:
        self.symbol = symbol.upper()
        self.client = http_client or httpx.AsyncClient(base_url=BINANCE_SPOT_BASE, timeout=30.0)
        self.rate_limiter = _RateLimiter(max_rate_per_sec=requests_per_second, burst=10)
        self.snapshot_limit = snapshot_limit

    async def iter_trades(self, start: datetime, end: datetime, *, limit: int = 1000) -> AsyncIterator[TradeTick]:
        start_ms = int(start.replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(end.replace(tzinfo=timezone.utc).timestamp() * 1000)
        last_id: Optional[int] = None

        while True:
            await self.rate_limiter.acquire()
            params: Dict[str, object] = {"symbol": self.symbol, "limit": limit}
            if last_id is not None:
                params["fromId"] = last_id + 1
            else:
                params["startTime"] = start_ms
                params["endTime"] = min(end_ms, start_ms + 60 * 60 * 1000)  # 1 hour chunks

            resp = await self.client.get("/api/v3/aggTrades", params=params)
            resp.raise_for_status()
            trades = resp.json()
            if not trades:
                if params.get("endTime", end_ms) >= end_ms:
                    break
                start_ms = int(params["endTime"])
                continue

            for raw in trades:
                ts = datetime.fromtimestamp(raw["T"] / 1000, tz=timezone.utc)
                if ts > end:
                    return
                last_id = int(raw["a"])
                yield TradeTick(
                    timestamp=ts,
                    price=float(raw["p"]),
                    quantity=float(raw["q"]),
                    is_buyer_maker=bool(raw["m"]),
                )

            if trades:
                end_cursor = trades[-1]["T"]
                if end_cursor >= end_ms:
                    break

    async def get_order_book(self, *, depth: int = 200) -> OrderBookSnapshot:
        depth = min(depth, self.snapshot_limit)
        await self.rate_limiter.acquire()
        resp = await self.client.get("/api/v3/depth", params={"symbol": self.symbol, "limit": depth})
        resp.raise_for_status()
        payload = resp.json()
        ts = datetime.fromtimestamp(payload["lastUpdateId"] / 1000, tz=timezone.utc)
        bids = [OrderBookLevel(float(px), float(qty)) for px, qty in payload["bids"]]
        asks = [OrderBookLevel(float(px), float(qty)) for px, qty in payload["asks"]]
        return OrderBookSnapshot(timestamp=ts, bids=bids, asks=asks)

    async def iter_order_book(
        self,
        start: datetime,
        end: datetime,
        *,
        step: timedelta = timedelta(seconds=1),
        depth: int = 200,
    ) -> AsyncIterator[OrderBookSnapshot]:
        current = start
        while current <= end:
            snapshot = await self.get_order_book(depth=depth)
            yield snapshot
            current += step
            await asyncio.sleep(step.total_seconds())


class BinanceFundingFetcher:
    """Fetch funding rates for perpetual futures symbols."""

    def __init__(
        self,
        *,
        symbol: str,
        http_client: Optional[httpx.AsyncClient] = None,
        requests_per_second: float = 5.0,
    ) -> None:
        self.symbol = symbol.upper()
        self.client = http_client or httpx.AsyncClient(base_url=BINANCE_FUTURES_BASE, timeout=30.0)
        self.rate_limiter = _RateLimiter(max_rate_per_sec=requests_per_second, burst=5)

    async def iter_funding_rates(self, start: datetime, end: datetime) -> AsyncIterator[FundingRate]:
        start_ms = int(start.replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(end.replace(tzinfo=timezone.utc).timestamp() * 1000)
        page_start = start_ms
        while page_start <= end_ms:
            await self.rate_limiter.acquire()
            resp = await self.client.get(
                "/fapi/v1/fundingRate",
                params={
                    "symbol": self.symbol,
                    "startTime": page_start,
                    "endTime": min(page_start + 7 * 24 * 60 * 60 * 1000, end_ms),
                    "limit": 1000,
                },
            )
            resp.raise_for_status()
            items = resp.json()
            if not items:
                page_start += 7 * 24 * 60 * 60 * 1000
                continue
            for raw in items:
                ts = datetime.fromtimestamp(raw["fundingTime"] / 1000, tz=timezone.utc)
                if ts > end:
                    return
                yield FundingRate(timestamp=ts, rate=float(raw["fundingRate"]), symbol=self.symbol)
            page_start = items[-1]["fundingTime"] + 1


class OnChainProvider:
    """Abstract base class for on-chain data providers."""

    async def iter_metrics(
        self, metric_names: Sequence[str], start: datetime, end: datetime
    ) -> AsyncIterator[OnChainMetric]:
        raise NotImplementedError


class NoopOnChainProvider(OnChainProvider):
    """Fallback provider that yields no metrics."""

    async def iter_metrics(
        self, metric_names: Sequence[str], start: datetime, end: datetime
    ) -> AsyncIterator[OnChainMetric]:
        if False:
            yield  # pragma: no cover
        return


class GlassnodeOnChainProvider(OnChainProvider):
    """Glassnode API wrapper. Requires API key."""

    BASE_URL = "https://api.glassnode.com/v1"

    def __init__(
        self,
        *,
        api_key: str,
        asset: str = "BTC",
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.api_key = api_key
        self.asset = asset.upper()
        self.client = http_client or httpx.AsyncClient(base_url=self.BASE_URL, timeout=30.0)

    async def iter_metrics(
        self, metric_names: Sequence[str], start: datetime, end: datetime
    ) -> AsyncIterator[OnChainMetric]:
        start_iso = start.replace(tzinfo=timezone.utc).isoformat()
        end_iso = end.replace(tzinfo=timezone.utc).isoformat()
        for metric in metric_names:
            resp = await self.client.get(
                f"/metrics/indicators/{metric}",
                params={"a": self.asset, "s": start_iso, "u": end_iso, "api_key": self.api_key},
            )
            resp.raise_for_status()
            series = resp.json()
            for point in series:
                ts = datetime.fromtimestamp(point["t"], tz=timezone.utc)
                if ts < start or ts > end:
                    continue
                yield OnChainMetric(timestamp=ts, name=metric, value=float(point["v"]))


async def gather_concurrently(tasks: Iterable[asyncio.Task]) -> List[object]:
    """Await a collection of asyncio tasks and propagate exceptions as a list."""
    results = await asyncio.gather(*tasks, return_exceptions=True)
    exceptions = [res for res in results if isinstance(res, Exception)]
    if exceptions:
        raise ExceptionGroup("one or more data fetchers failed", exceptions)
    return results

