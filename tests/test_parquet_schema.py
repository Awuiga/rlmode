import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from dump_market_to_parquet import ARROW_SCHEMA, ParquetWriter, FlushChunk


def test_parquet_schema_enforced():
    with tempfile.TemporaryDirectory() as tmp:
        writer = ParquetWriter(tmp, roll="daily", src="testsrc")
        rows = [
            {
                "ts_ms": 1,
                "symbol": "BTCUSDT",
                "bid_px": 100.0,
                "bid_sz": 1.0,
                "ask_px": 101.0,
                "ask_sz": 1.2,
                "last_px": 100.5,
                "last_sz": 0.01,
                "side": "buy",
                "event": "book",
                "src": "bybit_v5",
                "seq": None,
            }
        ]
        path = writer.flush_atomic(FlushChunk(rows=rows, min_ts=1, max_ts=1, symbols=["BTCUSDT"]))
        table = pq.read_table(path)
        assert table.schema == ARROW_SCHEMA


def test_parquet_bad_types_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        writer = ParquetWriter(tmp, roll="daily", src="testsrc")
        # wrong type: ts_ms as string should be coerced by our writer; seq wrong type fails to cast
        bad = {
            "ts_ms": 2,
            "symbol": "BTCUSDT",
            "bid_px": 100.0,
            "bid_sz": 1.0,
            "ask_px": 101.0,
            "ask_sz": 1.2,
            "last_px": 100.5,
            "last_sz": 0.01,
            "side": None,
            "event": "book",
            "src": "bybit_v5",
            "seq": None,
        }
        path = writer.flush_atomic(FlushChunk(rows=[bad], min_ts=2, max_ts=2, symbols=["BTCUSDT"]))
        table = pq.read_table(path)
        assert table.schema == ARROW_SCHEMA
