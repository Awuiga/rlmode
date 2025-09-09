from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import redis.asyncio as redis
from prometheus_client import Counter, Gauge, Histogram, start_http_server


# ---------------------- Logging (JSON-one-line) ----------------------

LOG = logging.getLogger("parquet_dumper")


def jlog(level: str, event: str, **fields: Any) -> None:
    payload = {"level": level, "ts": int(time.time() * 1000), "event": event}
    payload.update(fields)
    try:
        line = json.dumps(payload, ensure_ascii=False)
    except Exception:
        line = json.dumps({"level": level, "ts": payload["ts"], "event": event})
    print(line, flush=True)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO)


# ---------------------- Metrics ----------------------

PARQUET_FLUSH_ROWS = Counter("parquet_flush_rows_total", "Total rows flushed to parquet")
PARQUET_FLUSH_DURATION = Histogram(
    "parquet_flush_duration_ms", "Flush duration in milliseconds", buckets=(25, 50, 100, 250, 500, 1000, 2000, 5000)
)
REDIS_LAG = Gauge("redis_stream_lag_records", "Redis pending messages for consumer group")
REDIS_READ_BATCH = Histogram(
    "redis_read_batch_size", "Batch size of XREADGROUP", buckets=(1, 10, 50, 100, 250, 500, 1000)
)
CONSUMER_RESTARTS = Counter("consumer_restarts_total", "Consumer restarts")


# ---------------------- Schema ----------------------

SCHEMA_VERSION = "1"
ARROW_SCHEMA = pa.schema(
    [
        ("ts_ms", pa.int64()),
        ("symbol", pa.string()),
        ("bid_px", pa.float64()),
        ("bid_sz", pa.float64()),
        ("ask_px", pa.float64()),
        ("ask_sz", pa.float64()),
        ("last_px", pa.float64()),
        ("last_sz", pa.float64()),
        ("side", pa.string()),  # nullable
        ("event", pa.string()),  # e.g. 'book'/'trade'
        ("src", pa.string()),
        ("seq", pa.int64()),  # nullable
    ]
)


def _utc_ms() -> int:
    return int(time.time() * 1000)


def _to_int_ms(v: Any) -> int:
    try:
        iv = int(float(v))
        return iv * 1000 if iv < 10_000_000_000 else iv
    except Exception:
        return _utc_ms()


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def build_row(record: Mapping[str, Any], src: str, default_event: str = "book") -> Dict[str, Any]:
    # Expect project-native MarketEvent mappings
    ts_ms = _to_int_ms(record.get("ts", record.get("timestamp")))
    symbol = record.get("symbol") or record.get("sym")
    bid_px = _as_float(record.get("bid1"))
    ask_px = _as_float(record.get("ask1"))
    # Sizes from depth top-1 if available
    bid_sz = None
    ask_sz = None
    bids = record.get("bids") or []
    asks = record.get("asks") or []
    try:
        if isinstance(bids, list) and bids:
            bid_sz = _as_float(bids[0][1])
        if isinstance(asks, list) and asks:
            ask_sz = _as_float(asks[0][1])
    except Exception:
        pass
    last = record.get("last_trade") or {}
    last_px = _as_float(last.get("price")) if isinstance(last, Mapping) else None
    last_sz = _as_float(last.get("qty")) if isinstance(last, Mapping) else None
    side = last.get("side") if isinstance(last, Mapping) else None
    event = record.get("event") or default_event
    seq = record.get("seq")
    try:
        seq = int(seq) if seq is not None else None
    except Exception:
        seq = None
    return {
        "ts_ms": int(ts_ms),
        "symbol": str(symbol) if symbol is not None else None,
        "bid_px": bid_px,
        "bid_sz": bid_sz,
        "ask_px": ask_px,
        "ask_sz": ask_sz,
        "last_px": last_px,
        "last_sz": last_sz,
        "side": None if side is None else str(side),
        "event": str(event),
        "src": str(src),
        "seq": seq,
    }


class Deduper:
    def __init__(self, max_size: int = 200_000):
        from collections import deque

        self._set: set[Tuple[Any, Any, Any]] = set()
        self._dq: "deque[Tuple[Any, Any, Any]]" = deque()
        self._max = max_size

    def seen(self, symbol: Any, ts_ms: Any, last_px: Any) -> bool:
        key = (symbol, ts_ms, last_px)
        if key in self._set:
            return True
        self._set.add(key)
        self._dq.append(key)
        if len(self._dq) > self._max:
            old = self._dq.popleft()
            self._set.discard(old)
        return False


@dataclass
class FlushChunk:
    rows: List[Dict[str, Any]]
    min_ts: int
    max_ts: int
    symbols: List[str]


class ParquetWriter:
    def __init__(self, out_root: str, roll: str, src: str):
        self.out_root = out_root
        self.roll = roll  # 'hourly' | 'daily'
        self.src = src
        os.makedirs(self.out_root, exist_ok=True)

    def _partition_dir(self, ts_ms: int) -> str:
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        date_dir = os.path.join(self.out_root, f"date={dt.strftime('%Y-%m-%d')}")
        if self.roll == "hourly":
            return os.path.join(date_dir, f"hour={dt.strftime('%H')}")
        return date_dir

    def _choose_filename(self, min_ts: int, max_ts: int, rows: int) -> str:
        return f"{min_ts}-{max_ts}_{rows}.parquet"

    def _sidecar_path(self, parquet_path: str) -> str:
        root, _ = os.path.splitext(parquet_path)
        return root + ".json"

    def flush_atomic(self, chunk: FlushChunk) -> str:
        # Build DataFrame in fixed column order, enforce schema with pyarrow
        df = pd.DataFrame(chunk.rows, columns=[f.name for f in ARROW_SCHEMA])
        table = pa.Table.from_pandas(df, schema=ARROW_SCHEMA, preserve_index=False)
        # Paths
        out_dir = self._partition_dir(chunk.min_ts)
        os.makedirs(out_dir, exist_ok=True)
        fname = self._choose_filename(chunk.min_ts, chunk.max_ts, len(chunk.rows))
        final_path = os.path.join(out_dir, fname)
        # Atomic write: tmp file then replace
        with tempfile.NamedTemporaryFile(dir=out_dir, prefix=".tmp_parq_", suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            pq.write_table(table, tmp_path, compression="snappy")
            os.replace(tmp_path, final_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
        # Sidecar metadata
        sidecar = {
            "min_ts": chunk.min_ts,
            "max_ts": chunk.max_ts,
            "row_count": len(chunk.rows),
            "schema_ver": SCHEMA_VERSION,
            "src": self.src,
            "symbols": sorted(set(chunk.symbols)),
        }
        with open(self._sidecar_path(final_path), "w", encoding="utf-8") as f:
            json.dump(sidecar, f)
        return final_path


async def ensure_group(r: redis.Redis, stream: str, group: str) -> None:
    try:
        await r.xgroup_create(stream, group, id="$", mkstream=True)
        jlog("info", "xgroup_create", stream=stream, group=group)
    except Exception as e:
        msg = str(e)
        if "BUSYGROUP" in msg:
            return
        jlog("error", "xgroup_create_failed", stream=stream, group=group, error=msg)


async def get_group_lag(r: redis.Redis, stream: str, group: str) -> int:
    try:
        info = await r.xpending(stream, group)
        # Returns PendingInfo (count, min, max, consumers). We care about count.
        return int(info[0]) if isinstance(info, (list, tuple)) and info else 0
    except Exception:
        return 0


async def consume(
    r: redis.Redis,
    stream: str,
    group: str,
    consumer: str,
    out_dir: str,
    roll: str,
    src: str,
    max_batch: int,
    flush_seconds: int,
    idle_ms: int,
    stop_event: asyncio.Event,
) -> None:
    writer = ParquetWriter(out_dir, roll=roll, src=src)
    dedup = Deduper()
    last_flush_t = time.monotonic()
    to_write: List[Dict[str, Any]] = []
    ack_ids: List[str] = []
    symbols_accum: List[str] = []
    min_ts = 2**63 - 1
    max_ts = 0
    await ensure_group(r, stream, group)
    jlog("info", "consumer_started", stream=stream, group=group, consumer=consumer)

    cur_max_batch = max_batch
    cur_idle_ms = idle_ms
    LAG_HIGH = 10000
    LAG_LOW = 1000
    while not stop_event.is_set():
        try:
            lag = await get_group_lag(r, stream, group)
            REDIS_LAG.set(lag)
            # Simple latency guards: adapt batch/idle based on lag
            if lag > LAG_HIGH:
                cur_max_batch = max(1000, max_batch // 2)
                cur_idle_ms = min(2000, idle_ms * 2)
            elif lag < LAG_LOW:
                cur_max_batch = max_batch
                cur_idle_ms = idle_ms

            resp = await r.xreadgroup(group, consumer, streams={stream: ">"}, count=cur_max_batch, block=cur_idle_ms)
        except Exception as e:
            CONSUMER_RESTARTS.inc()
            jlog("error", "xreadgroup_failed", error=str(e))
            await asyncio.sleep(1.0)
            continue

        batch_size = 0
        if resp:
            # resp is list of (stream, [(id, data), ...])
            for _s, msgs in resp:
                batch_size += len(msgs)
                for msg_id, data in msgs:
                    try:
                        record = data if isinstance(data, Mapping) else {}
                        row = build_row(record, src=src)
                        if dedup.seen(row["symbol"], row["ts_ms"], row["last_px"]):
                            continue
                        to_write.append(row)
                        ack_ids.append(msg_id)
                        symbols_accum.append(row["symbol"])  # type: ignore[arg-type]
                        ts = int(row["ts_ms"]) if row["ts_ms"] is not None else _utc_ms()
                        if ts < min_ts:
                            min_ts = ts
                        if ts > max_ts:
                            max_ts = ts
                    except Exception as e:
                        jlog("error", "parse_failed", error=str(e))
        REDIS_READ_BATCH.observe(batch_size)

        # Flush by time or size
        now = time.monotonic()
        do_flush = (len(to_write) >= max_batch) or ((now - last_flush_t) >= flush_seconds)
        if do_flush and to_write:
            t0 = time.perf_counter()
            try:
                chunk = FlushChunk(rows=to_write, min_ts=min_ts, max_ts=max_ts, symbols=symbols_accum)
                path = writer.flush_atomic(chunk)
                await r.xack(stream, group, *ack_ids)
                dur_ms = int((time.perf_counter() - t0) * 1000)
                PARQUET_FLUSH_ROWS.inc(len(to_write))
                PARQUET_FLUSH_DURATION.observe(dur_ms)
                jlog(
                    "info",
                    "parquet_flush",
                    rows=len(to_write),
                    path=path,
                    lag_ms=max(0, _utc_ms() - max_ts),
                    redis_lag=int(REDIS_LAG._value.get()),  # type: ignore[attr-defined]
                )
            except Exception as e:
                jlog("error", "flush_failed", error=str(e))
            finally:
                to_write = []
                ack_ids = []
                symbols_accum = []
                min_ts = 2**63 - 1
                max_ts = 0
                last_flush_t = now

        # Heartbeat when idle
        if not resp and (now - last_flush_t) >= max(1, flush_seconds // 2):
            jlog("info", "heartbeat", redis_lag=int(REDIS_LAG._value.get()))  # type: ignore[attr-defined]


async def main_async(args: argparse.Namespace) -> None:
    # Metrics HTTP server
    if args.metrics_port:
        start_http_server(args.metrics_port, addr=args.metrics_host)
        jlog("info", "metrics_http_started", host=args.metrics_host, port=args.metrics_port)

    r = redis.from_url(args.redis, decode_responses=True)
    stop_event = asyncio.Event()

    def _graceful() -> None:
        stop_event.set()
        jlog("info", "shutdown_signal")

    loop = asyncio.get_running_loop()
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _graceful)
        except Exception:
            pass

    await consume(
        r=r,
        stream=args.stream,
        group=args.group,
        consumer=args.consumer or f"{socket.gethostname()}-{os.getpid()}",
        out_dir=args.out,
        roll=args.roll,
        src=args.src,
        max_batch=args.max_batch,
        flush_seconds=args.flush_seconds,
        idle_ms=args.idle_ms,
        stop_event=stop_event,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump Redis Stream market data to partitioned Parquet files")
    p.add_argument("--redis", default="redis://localhost:6379/0", help="Redis URL")
    p.add_argument("--stream", default="md:raw", help="Redis stream name")
    p.add_argument("--out", default="./data/parquet", help="Output root directory")
    p.add_argument("--roll", choices=["hourly", "daily"], default="hourly", help="Partition roll up")
    p.add_argument("--max-batch", dest="max_batch", type=int, default=25_000, help="Max rows per flush")
    p.add_argument("--idle-ms", dest="idle_ms", type=int, default=500, help="XREADGROUP block (ms)")
    p.add_argument("--group", default="parquet:grp", help="Consumer group name")
    p.add_argument("--consumer", default=None, help="Consumer name (default hostname-pid)")
    p.add_argument("--flush-seconds", dest="flush_seconds", type=int, default=10, help="Flush interval seconds")
    p.add_argument("--src", default="unknown", help="Source tag: bybit_v5 | binance_futs | ...")
    p.add_argument("--metrics-host", default="0.0.0.0", help="Metrics HTTP host")
    p.add_argument("--metrics-port", type=int, default=8001, help="Metrics HTTP port")
    return p.parse_args(argv)


def main() -> None:
    setup_logging()
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
