import asyncio
import json
import logging
from typing import List

import redis.asyncio as redis
import yaml
import requests

LOGGER = logging.getLogger("executor")


class ExchangeAPI:
    def __init__(self, base_url: str, key: str, secret: str):
        self.base_url = base_url
        self.key = key
        self.secret = secret

    def place_order(self, side: str, price: float, qty: float, post_only: bool = True) -> dict:
        LOGGER.info("placing %s order price=%s qty=%s", side, price, qty)
        return {"id": "stub"}

    def cancel(self, order_id: str) -> None:
        LOGGER.info("cancel %s", order_id)


class Executor:
    """Place maker orders for confirmed signals and manage stops."""

    def __init__(self, config_path: str = "config.yml", redis_url: str = "redis://localhost:6379/0"):
        with open(config_path) as fh:
            self.cfg = yaml.safe_load(fh)
        self.redis = redis.from_url(redis_url)
        self.api = ExchangeAPI(self.cfg.get("api_url", ""), "", "")

    async def run(self):
        while True:
            raw = await self.redis.brpop("signals")
            sig = json.loads(raw[1])
            await self.execute(sig)

    async def execute(self, sig: dict):
        entry = sig["entry"]
        ladder = self.cfg.get("ladder", [1.0])
        qty = self.cfg.get("qty", 0.001)
        sigma = sig.get("features", {}).get("sigma", 0)
        stop_price = entry - max(entry * self.cfg.get("min_stop", 0.0006), self.cfg.get("adaptive_stop_k", 1.2) * sigma)
        for i, pct in enumerate(ladder):
            price = entry - i * self.cfg.get("ladder_step", 0.1)
            self.api.place_order(sig["side"], price, qty * pct, post_only=True)
        LOGGER.info("placed ladder, stop=%s", stop_price)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    ex = Executor()
    asyncio.run(ex.run())
