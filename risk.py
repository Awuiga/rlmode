import asyncio
import json
import logging

import redis.asyncio as redis
import yaml

LOGGER = logging.getLogger("risk")


class RiskManager:
    """Track PnL and stop trading on drawdowns or loss streaks."""

    def __init__(self, risk_path: str = "risk.yml", redis_url: str = "redis://localhost:6379/0"):
        with open(risk_path) as fh:
            self.cfg = yaml.safe_load(fh)
        self.redis = redis.from_url(redis_url)
        self.daily_pnl = 0.0
        self.loss_streak = 0

    async def run(self):
        while True:
            raw = await self.redis.brpop("fills")
            fill = json.loads(raw[1])
            pnl = fill.get("pnl", 0.0)
            self.daily_pnl += pnl
            if pnl < 0:
                self.loss_streak += 1
            else:
                self.loss_streak = 0
            if self.daily_pnl <= self.cfg.get("daily_loss_limit", -0.03) or self.loss_streak >= self.cfg.get("max_consecutive_losses", 5):
                LOGGER.warning("risk limits triggered, stopping")
                await self.redis.lpush("control", "stop")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    rm = RiskManager()
    asyncio.run(rm.run())
