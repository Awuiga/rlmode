import asyncio
import json
import logging
from dataclasses import dataclass
from typing import List

import websockets
import redis.asyncio as redis

LOGGER = logging.getLogger("collector")


@dataclass
class MarketData:
    book: dict
    trades: List[dict]


class Collector:
    """Subscribe to depth and trades and push raw events to Redis."""

    def __init__(self, exchange: str, symbol: str, redis_url: str = "redis://localhost:6379/0"):
        self.exchange = exchange.lower()
        self.symbol = symbol.lower()
        self.redis = redis.from_url(redis_url)

    def _ws_url(self) -> str:
        if self.exchange == "binance":
            return f"wss://stream.binance.com:9443/stream?streams={self.symbol}@depth20@100ms/{self.symbol}@trade"
        if self.exchange == "bybit":
            return f"wss://stream.bybit.com/contract/usdt/public/v3"
        raise ValueError("Unsupported exchange")

    async def run(self):
        url = self._ws_url()
        while True:
            try:
                async with websockets.connect(url) as ws:
                    LOGGER.info("connected to %s", url)
                    async for raw in ws:
                        await self.redis.lpush("raw", raw)
            except Exception as exc:  # pragma: no cover - network specific
                LOGGER.exception("collector error: %s", exc)
                await asyncio.sleep(5)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    cfg_exchange = "binance"
    cfg_symbol = "btcusdt"
    collector = Collector(cfg_exchange, cfg_symbol)
    asyncio.run(collector.run())
