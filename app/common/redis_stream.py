from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Mapping

import json

import redis

try:
    import fakeredis
except Exception:  # pragma: no cover
    fakeredis = None

from pydantic import BaseModel
from dataclasses import is_dataclass, asdict
from .logging import get_logger


log = get_logger("redis_stream")


class RedisStream:
    def __init__(self, url: str, *, default_maxlen: Optional[int] = None):
        self.url = url
        self.default_maxlen = default_maxlen
        if url.startswith("fakeredis://"):
            if fakeredis is None:
                raise RuntimeError("fakeredis not installed")
            self.client = fakeredis.FakeRedis(decode_responses=True)
        else:
            self.client = redis.Redis.from_url(url, decode_responses=True)

    def set_default_maxlen(self, maxlen: Optional[int]) -> None:
        self.default_maxlen = maxlen

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

    def xadd(
        self,
        stream: str,
        data: Any,
        idempotency_key: Optional[str] = None,
        *,
        maxlen: int | None = None,
    ) -> Optional[str]:
        # Normalize input into a plain dict, drop None, JSON-serialize complex values
        payload: Dict[str, Any]
        if isinstance(data, BaseModel):
            try:
                payload = data.model_dump(mode="json")  # pydantic v2 preferred
            except TypeError:
                payload = data.model_dump()
        elif hasattr(data, "dict") and callable(getattr(data, "dict")):
            payload = dict(data.dict())  # pydantic v1 or similar
        elif is_dataclass(data):
            payload = asdict(data)
        elif isinstance(data, Mapping):
            payload = dict(data)
        else:
            raise TypeError(f"xadd unsupported payload type: {type(data)!r}")

        cleaned: Dict[str, Any] = {}
        for k, v in payload.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bytes)):
                cleaned[k] = v
            else:
                cleaned[k] = json.dumps(v, separators=(",", ":"))

        if idempotency_key:
            key = f"idem:{stream}:{idempotency_key}"
            # Avoid duplicates within 24h
            if not self.client.set(name=key, value="1", nx=True, ex=86400):
                # Duplicate, ignore
                log.info("xadd_skip_duplicate", stream=stream, idem=key)
                return None

        use_maxlen = maxlen if maxlen is not None else self.default_maxlen
        return self.client.xadd(stream, cleaned, maxlen=use_maxlen)

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
