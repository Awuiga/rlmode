import asyncio
import json
import logging
import os
from typing import Dict

import redis.asyncio as redis
from prometheus_client import start_http_server, Gauge
import requests

LOGGER = logging.getLogger("monitor")


class Monitor:
    def __init__(self, redis_url: str = "redis://localhost:6379/0", metrics_port: int = 8000):
        self.redis = redis.from_url(redis_url)
        self.metrics: Dict[str, Gauge] = {}
        start_http_server(metrics_port)
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT")

    async def run(self):
        while True:
            raw = await self.redis.brpop("metrics")
            metric = json.loads(raw[1])
            name = metric["name"]
            value = metric["value"]
            if name not in self.metrics:
                self.metrics[name] = Gauge(name, name)
            self.metrics[name].set(value)
            if metric.get("alert"):
                self._send_telegram(f"{name}: {value}")

    def _send_telegram(self, text: str):
        if not (self.token and self.chat_id):
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": self.chat_id, "text": text})
        except Exception as exc:  # pragma: no cover - network
            LOGGER.error("telegram error: %s", exc)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    mon = Monitor()
    asyncio.run(mon.run())
