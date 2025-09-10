from __future__ import annotations

import os
from typing import Dict, Tuple

import httpx
from prometheus_client import start_http_server, Counter, Gauge

from ..common.config import load_app_config
from ..common.logging import setup_logging, get_logger
from ..common.redis_stream import RedisStream
from ..common.schema import Metric, ControlEvent


log = get_logger("monitor")


class MetricRegistry:
    def __init__(self):
        self.counters: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Counter] = {}
        self.gauges: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Gauge] = {}

    def _key(self, name: str, labels: Dict[str, str] | None) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
        items = tuple(sorted((labels or {}).items()))
        return name, items

    def inc(self, name: str, value: float, labels: Dict[str, str] | None = None):
        key = self._key(name, labels)
        if key not in self.counters:
            label_names = [k for k, _ in key[1]]
            self.counters[key] = Counter(name, name, labelnames=label_names)
        counter = self.counters[key]
        counter.labels(**dict(key[1])).inc(value)

    def set(self, name: str, value: float, labels: Dict[str, str] | None = None):
        key = self._key(name, labels)
        if key not in self.gauges:
            label_names = [k for k, _ in key[1]]
            self.gauges[key] = Gauge(name, name, labelnames=label_names)
        gauge = self.gauges[key]
        gauge.labels(**dict(key[1])).set(value)


async def send_telegram(token: str, chat_id: str, text: str):
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": text})
    except Exception as e:
        log.warning("telegram_failed", error=str(e))


def main():
    setup_logging()
    cfg = load_app_config()
    rs = RedisStream(cfg.redis.url)

    # Start HTTP server
    start_http_server(cfg.monitor.http_port, addr=cfg.monitor.http_host)
    log.info("monitor_http_started", host=cfg.monitor.http_host, port=cfg.monitor.http_port)

    metrics = MetricRegistry()
    tg_token = os.environ.get("TELEGRAM_TOKEN")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID")

    group = "monitor"
    consumer = "c1"
    streams = [
        "metrics:collector",
        "metrics:signal",
        "metrics:ai",
        "metrics:executor",
        "metrics:risk",
        "metrics:parquet",
        "control:events",
    ]

    while True:
        msgs = rs.read_group(group=group, consumer=consumer, streams=streams, block_ms=1000, count=100)
        if not msgs:
            continue
        for stream, items in msgs:
            for msg_id, data in items:
                try:
                    if stream.startswith("metrics:"):
                        m = Metric.model_validate(data)
                        if m.name.endswith("_total"):
                            metrics.inc(m.name, m.value, m.labels)
                        else:
                            metrics.set(m.name, m.value, m.labels)
                    elif stream == "control:events":
                        evt = ControlEvent.model_validate(data)
                        log.error("control_event", type=evt.type, reason=evt.reason)
                        if tg_token and tg_chat:
                            # Send async but fire-and-forget; network may be restricted
                            try:
                                import asyncio

                                asyncio.get_event_loop().create_task(
                                    send_telegram(tg_token, tg_chat, f"[{evt.type}] {evt.reason}")
                                )
                            except Exception as e:
                                log.warning("telegram_schedule_failed", error=str(e))
                except Exception as e:
                    log.error("monitor_error", error=str(e))
                finally:
                    rs.ack(stream, group, msg_id)


if __name__ == "__main__":
    main()
