from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:
    import orjson as _json  # type: ignore
except Exception:  # pragma: no cover
    import json as _json  # type: ignore

import pandas as pd
import redis.asyncio as redis


LOG = logging.getLogger("dump_market")


def _utc_ms() -> int:
    return int(time.time() * 1000)


def _looks_like_json(s: str) -> bool:
    s = s.strip()
    return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))


def _to_int_ms(v: Any) -> int:
    try:
        iv = int(float(v))
        # Heuristic: if value looks like seconds (<= 10^10), convert to ms
        return iv * 1000 if iv < 10_000_000_000 else iv
    except Exception:
        return _utc_ms()


def _safe_json_dumps(obj: Any) -> str:
    try:
        return _json.dumps(obj).decode("utf-8") if hasattr(_json, "dumps") else _json.dumps(obj)
    except Exception:
        try:
            import json as _fallback

            return _fallback.dumps(obj)
        except Exception:
            return str(obj)


def _safe_json_loads(s: str) -> Any:
    try:
        return _json.loads(s)  # type: ignore[attr-defined]
    except Exception:
        try:
            import json as _fallback

            return _fallback.loads(s)
        except Exception:
            return s


def _canonical_row(
    record: Mapping[str, Any],
    stream_id: str,
) -> Dict[str, Any]:
    # If record is a wrapper containing a single JSON payload field, unwrap
    if len(record) == 1:
        k = next(iter(record))
        v = record[k]
        if isinstance(v, (bytes, bytearray)):
            try:
                v = v.decode("utf-8")
            except Exception:
                pass
        if isinstance(v, str) and _looks_like_json(v):
            try:
                parsed = _safe_json_loads(v)
                if isinstance(parsed, Mapping):
                    record = parsed  # type: ignore[assignment]
            except Exception:
                pass

    # Flatten supported fields and keep extras minimal
    row: Dict[str, Any] = {}
    # Timestamp
    ts_val = record.get("ts") or record.get("timestamp")
    row["ts"] = _to_int_ms(ts_val) if ts_val is not None else _utc_ms()
    # Symbol
    sym = record.get("symbol") or record.get("sym")
    if isinstance(sym, (bytes, bytearray)):
        try:
            sym = sym.decode("utf-8")
        except Exception:
            pass
    row["symbol"] = sym
    # Top of book
    for f in ("bid1", "ask1"):
        val = record.get(f)
        try:
            row[f] = float(val) if val is not None else None
        except Exception:
            row[f] = None
    # Depth/last trade (store as JSON string for compactness and schema stability)
    for f in ("bids", "asks"):
        v = record.get(f)
        if isinstance(v, str) and _looks_like_json(v):
            row[f] = v
        elif isinstance(v, (list, tuple)):
            row[f] = _safe_json_dumps(v)
        else:
            # best-effort parse, otherwise keep as string
            try:
                row[f] = _safe_json_dumps(_safe_json_loads(v)) if isinstance(v, str) else (str(v) if v is not None else None)
            except Exception:
                row[f] = str(v) if v is not None else None

    lt = record.get("last_trade")
    if isinstance(lt, str) and _looks_like_json(lt):
        try:
            lt = _safe_json_loads(lt)
        except Exception:
            pass
    if isinstance(lt, Mapping):
        row["last_trade_price"] = (
            float(lt.get("price")) if lt.get("price") is not None else None
        )
        row["last_trade_qty"] = (
            float(lt.get("qty")) if lt.get("qty") is not None else None
        )
        side = lt.get("side")
        if isinstance(side, (bytes, bytearray)):
            try:
                side = side.decode("utf-8")
            except Exception:
                pass
        row["last_trade_side"] = side
    else:
        row["last_trade_price"] = None
        row["last_trade_qty"] = None
        row["last_trade_side"] = None

    # Keep stream id for traceability
    row["stream_id"] = stream_id
    return row


class ParquetSink:
    def __init__(self, out_dir: str, engine: str = "pyarrow") -> None:
        self.out_dir = out_dir
        self.engine = engine
        self._writers: Dict[str, Any] = {}  # hour_key -> writer (pyarrow)
        self._schemas: Dict[str, Any] = {}
        self._paths: Dict[str, str] = {}
        if engine not in ("pyarrow", "fastparquet"):
            raise ValueError("engine must be 'pyarrow' or 'fastparquet'")
        if engine == "pyarrow":
            try:
                import pyarrow as pa  # noqa: F401
                import pyarrow.parquet as pq  # noqa: F401
            except Exception as e:  # pragma: no cover
                raise RuntimeError("pyarrow is required for engine='pyarrow'") from e

    def _file_path(self, hour_key: str) -> str:
        return os.path.join(self.out_dir, f"{hour_key}.parquet")

    def _unique_path(self, base_path: str) -> str:
        if not os.path.exists(base_path):
            return base_path
        root, ext = os.path.splitext(base_path)
        i = 1
        while True:
            cand = f"{root}_{i}{ext}"
            if not os.path.exists(cand):
                return cand
            i += 1

    def _ensure_dir(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)

    def write(self, hour_key: str, df: pd.DataFrame) -> None:
        self._ensure_dir()
        # Resolve target path
        path = self._paths.get(hour_key)
        if not path:
            base = self._file_path(hour_key)
            if self.engine == "pyarrow":
                # Avoid clobbering existing files across restarts
                path = self._unique_path(base)
            else:
                path = base
            self._paths[hour_key] = path
        if self.engine == "fastparquet":
            append = os.path.exists(path)
            df.to_parquet(path, engine="fastparquet", index=False, compression="snappy", append=append)
            LOG.info("saved_parquet path=%s rows=%d", path, len(df))
            return

        # pyarrow: keep a writer open per hour_key for appends within process lifetime
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df)
        writer = self._writers.get(hour_key)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression="snappy")
            self._writers[hour_key] = writer
            self._schemas[hour_key] = table.schema
        else:
            # Align to schema if columns drift
            schema = self._schemas[hour_key]
            if table.schema != schema:
                # Reorder/add missing columns as nulls
                cols = [c.name for c in schema]
                for c in cols:
                    if c not in df.columns:
                        df[c] = None
                df = df[cols]
                table = pa.Table.from_pandas(df)
        writer.write_table(table)
        LOG.info("saved_parquet path=%s rows=%d", path, len(df))

    def close(self) -> None:
        if self.engine == "pyarrow":
            for k, w in list(self._writers.items()):
                try:
                    w.close()
                except Exception:  # pragma: no cover
                    pass
            self._writers.clear()
            self._schemas.clear()


async def _read_stream(
    r: redis.Redis,
    stream: str,
    start_id: str,
    buffer_max: int,
    flush_interval_sec: int,
    out_dir: str,
    engine: str,
    block_ms: int,
    stop_event: asyncio.Event,
) -> None:
    last_id = start_id
    buffers: Dict[str, List[Dict[str, Any]]] = {}  # hour_key -> rows
    sink = ParquetSink(out_dir, engine=engine)
    last_flush = time.monotonic()

    async def flush(reason: str = "periodic") -> None:
        nonlocal buffers, last_flush
        if not any(buffers.values()):
            last_flush = time.monotonic()
            return
        for hour_key, rows in list(buffers.items()):
            if not rows:
                continue
            df = pd.DataFrame(rows)
            try:
                sink.write(hour_key, df)
            except Exception:
                LOG.exception("parquet_write_failed hour=%s", hour_key)
            buffers[hour_key] = []
        last_flush = time.monotonic()

    try:
        while not stop_event.is_set():
            try:
                resp = await r.xread({stream: last_id}, block=block_ms, count=100)
            except Exception:
                LOG.exception("xread_failed")
                await asyncio.sleep(1.0)
                continue

            if not resp:
                # Check time-based flush
                if time.monotonic() - last_flush >= flush_interval_sec:
                    await flush("interval")
                continue

            for _stream_name, messages in resp:
                for msg_id, data in messages:
                    # data is Mapping[str, str]; convert and normalize
                    try:
                        record = data if isinstance(data, Mapping) else {}
                        row = _canonical_row(record, stream_id=msg_id)
                    except Exception:
                        LOG.exception("parse_failed")
                        continue

                    # Decide hour by event ts (UTC)
                    dt = datetime.fromtimestamp((row.get("ts") or _utc_ms()) / 1000.0, tz=timezone.utc)
                    hour_key = dt.strftime("%Y%m%d_%H")
                    buflist = buffers.setdefault(hour_key, [])
                    buflist.append(row)

                    # Batching flush
                    total = sum(len(v) for v in buffers.values())
                    if total >= buffer_max:
                        await flush("batch")
                # Advance last_id to the last message id we saw from this read
                if messages:
                    last_id = messages[-1][0]

            # Interval flush check
            if time.monotonic() - last_flush >= flush_interval_sec:
                await flush("interval")
    finally:
        try:
            await flush("shutdown")
        finally:
            sink.close()


async def main_async(args: argparse.Namespace) -> None:
    r = redis.from_url(args.redis, decode_responses=True)

    stop_event = asyncio.Event()

    def _graceful() -> None:
        LOG.info("shutdown_signal_received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    # Signal handlers (best-effort on Windows)
    try:
        loop.add_signal_handler(signal.SIGINT, _graceful)
    except (NotImplementedError, RuntimeError):  # pragma: no cover
        pass
    try:
        loop.add_signal_handler(signal.SIGTERM, _graceful)
    except (AttributeError, NotImplementedError, RuntimeError):  # pragma: no cover
        pass

    start_id = "$" if args.from_latest else "0-0"
    await _read_stream(
        r=r,
        stream=args.stream,
        start_id=start_id,
        buffer_max=args.batch,
        flush_interval_sec=args.interval,
        out_dir=args.out,
        engine=args.engine,
        block_ms=args.block_ms,
        stop_event=stop_event,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump market stream to hourly Parquet files")
    p.add_argument("--redis", default="redis://localhost:6379/0", help="Redis URL")
    p.add_argument("--stream", default="md:raw", help="Redis stream name")
    p.add_argument("--batch", type=int, default=10_000, help="Flush after N records")
    p.add_argument("--interval", type=int, default=60, help="Flush interval seconds")
    p.add_argument("--out", default="./data/market", help="Output directory")
    p.add_argument("--engine", choices=["pyarrow", "fastparquet"], default="pyarrow", help="Parquet engine")
    p.add_argument("--from-latest", dest="from_latest", action="store_true", help="Start from latest ($)")
    p.add_argument("--from-start", dest="from_latest", action="store_false", help="Start from 0-0")
    p.add_argument("--block-ms", type=int, default=1000, help="XREAD block timeout in ms")
    p.set_defaults(from_latest=True)
    return p.parse_args(argv)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    setup_logging()
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:  # pragma: no cover
        LOG.info("keyboard_interrupt")


if __name__ == "__main__":
    main()
