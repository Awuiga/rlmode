import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Dict, Any

import redis.asyncio as redis
import yaml

LOGGER = logging.getLogger("signals")


@dataclass
class SignalCandidate:
    side: str
    entry: float
    tp: float
    sl: float
    score: float
    features: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.__dict__)


class SignalEngine:
    """Rule based filter that consumes market data and emits trade candidates."""

    def __init__(self, config_path: str = "config.yml", redis_url: str = "redis://localhost:6379/0"):
        with open(config_path) as fh:
            self.cfg = yaml.safe_load(fh)
        self.redis = redis.from_url(redis_url)

    async def run(self):
        while True:
            raw = await self.redis.brpop("raw")
            event = json.loads(raw[1])
            sig = self._check_rules(event)
            if sig:
                await self.redis.lpush("candidates", sig.to_json())

    def _check_rules(self, event: Dict[str, Any]) -> SignalCandidate | None:
        # placeholder: compute basic spread and use threshold
        book = event.get("data", {}).get("b", [])
        if not book:
            return None
        bid = float(book[0][0])
        ask = float(book[0][0])
        spread = ask - bid
        if spread > self.cfg.get("max_spread", 5.0):
            return None
        entry = (ask + bid) / 2
        tp = entry * (1 + self.cfg["tp"])
        sl = entry * (1 - self.cfg["sl"])
        return SignalCandidate("long", entry, tp, sl, score=1.0, features={"spread": spread})


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    engine = SignalEngine()
    asyncio.run(engine.run())
