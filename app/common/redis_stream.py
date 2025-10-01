from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Mapping

import json
import time

import redis
from redis.exceptions import RedisError

try:
    import fakeredis
except Exception:  # pragma: no cover
    fakeredis = None

from pydantic import BaseModel
from dataclasses import is_dataclass, asdict
from .logging import get_logger


log = get_logger("redis_stream")


class RedisStream:
    def __init__(
        self,
        url: str,
        *,
        default_maxlen: Optional[int] = None,
        pending_retry: Optional[Any] = None,
    ):
        self.url = url
        self.default_maxlen = default_maxlen
        self._pending_retry = pending_retry
        self._backoff_index = 0
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
        try:
            result = self.client.xreadgroup(group, consumer, stream_dict, count=count, block=block_ms)
            self._backoff_index = 0
        except RedisError as exc:
            log.warning("xreadgroup_error", error=str(exc))
            delay = 0.0
            if self._pending_retry:
                backoff = getattr(self._pending_retry, "backoff_ms", None) or []
                if backoff:
                    idx = min(self._backoff_index, len(backoff) - 1)
                    delay = max(backoff[idx], 0) / 1000.0
                    self._backoff_index = min(idx + 1, len(backoff) - 1)
            if delay > 0:
                time.sleep(delay)
            return []

        if result:
            return result

        # No new messages; attempt to claim stale pending entries
        if not self._pending_retry:
            return []
        idle_ms = max(int(getattr(self._pending_retry, "idle_ms", 0) or 0), 0)
        if idle_ms <= 0:
            return []
        claimed_total: List[Tuple[str, List[Tuple[str, Dict[str, Any]]]]] = []
        claim_count = int(getattr(self._pending_retry, "count", 0) or 0)
        for stream in names:
            try:
                next_id = "0-0"
                messages: List[Tuple[str, Dict[str, Any]]] = []
                remaining = claim_count if claim_count > 0 else None
                while True:
                    res = self.client.xautoclaim(
                        stream,
                        group,
                        consumer,
                        min_idle_time=idle_ms,
                        start_id=next_id,
                        count=remaining,
                    )
                    if not res:
                        break
                    next_id, batch = res
                    if not batch:
                        break
                    messages.extend(batch)
                    if remaining is not None:
                        remaining -= len(batch)
                        if remaining <= 0:
                            break
                if messages:
                    claimed_total.append((stream, messages))
            except RedisError as exc:
                log.warning("xautoclaim_error", stream=stream, error=str(exc))
        return claimed_total

    def ack(self, stream: str, group: str, message_id: str) -> int:
        return self.client.xack(stream, group, message_id)

    def pending_length(self, stream: str, group: str) -> int:
        try:
            info = self.client.xpending(stream, group)
        except RedisError:
            return 0
        if isinstance(info, dict):
            return int(info.get("pending", 0) or 0)
        try:
            return int(info)
        except (TypeError, ValueError):
            return 0

    def stream_length(self, stream: str) -> int:
        try:
            return int(self.client.xlen(stream))
        except RedisError:
            return 0
