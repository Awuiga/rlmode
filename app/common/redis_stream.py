from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import redis

try:
    import fakeredis
except Exception:  # pragma: no cover
    fakeredis = None

from pydantic import BaseModel
from .logging import get_logger


log = get_logger("redis_stream")


class RedisStream:
    def __init__(self, url: str):
        self.url = url
        if url.startswith("fakeredis://"):
            if fakeredis is None:
                raise RuntimeError("fakeredis not installed")
            self.client = fakeredis.FakeRedis(decode_responses=True)
        else:
            self.client = redis.Redis.from_url(url, decode_responses=True)

    def _ensure_group(self, stream: str, group: str) -> None:
        try:
            self.client.xgroup_create(stream, group, id="$", mkstream=True)
            log.info("created_consumer_group", stream=stream, group=group)
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                return
            # Stream may be empty; mkstream ensures stream exists
            if "NOGROUP" in str(e):
                return
            # Otherwise log and continue
            log.warning("xgroup_create_error", stream=stream, group=group, error=str(e))

    def ensure_group(self, stream: str, group: str) -> None:
        self._ensure_group(stream, group)

    def xadd(self, stream: str, data: BaseModel, idempotency_key: Optional[str] = None) -> Optional[str]:
        payload = data.model_dump(mode="json")
        idem_id: Optional[str] = None
        if idempotency_key:
            key = f"idem:{stream}:{idempotency_key}"
            # Avoid duplicates within 24h
            if self.client.set(name=key, value="1", nx=True, ex=86400):
                pass
            else:
                # Duplicate, ignore
                log.info("xadd_skip_duplicate", stream=stream, idem=key)
                return None
        msg_id = self.client.xadd(stream, payload)
        return msg_id

    def read_group(
        self,
        group: str,
        consumer: str,
        streams,
        block_ms: int = 1000,
        count: int = 10,
    ) -> List[Tuple[str, List[Tuple[str, Dict[str, Any]]]]]:
        # Accept either list of stream names or explicit dict of last IDs
        if isinstance(streams, dict):
            names = list(streams.keys())
            stream_dict = dict(streams)
        else:
            names = list(streams)
            stream_dict = {s: ">" for s in names}
        # Ensure groups exist for all streams
        for s in names:
            self._ensure_group(s, group)
        result = self.client.xreadgroup(group, consumer, stream_dict, count=count, block=block_ms)
        return result or []

    def ack(self, stream: str, group: str, message_id: str) -> int:
        return self.client.xack(stream, group, message_id)
