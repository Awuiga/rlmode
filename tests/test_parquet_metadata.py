from __future__ import annotations

import json
from pathlib import Path

import pytest
import pyarrow as pa
pytestmark = pytest.mark.unit

import pyarrow.parquet as pq

from app.common.config import FeatureConfig
from dump_market_to_parquet import (
    ARROW_SCHEMA,
    FEATURE_CONTEXT,
    FlushChunk,
    ParquetWriter,
    init_feature_context,
)


def _blank_row() -> dict:
    row: dict[str, object] = {}
    for field in ARROW_SCHEMA:
        name = field.name
        if pa.types.is_integer(field.type):
            row[name] = 0
        elif pa.types.is_floating(field.type):
            row[name] = 0.0
        elif pa.types.is_string(field.type):
            row[name] = ""
        else:
            row[name] = None
    row["ts_ms"] = 1_000_000
    row["symbol"] = "BTCUSDT"
    row["event"] = "book"
    row["side"] = "BID"
    row["src"] = "unit-test"
    row["seq"] = 1
    return row


def test_parquet_writer_embeds_feature_schema_metadata(tmp_path):
    cfg = FeatureConfig()
    init_feature_context(cfg)

    row = _blank_row()
    row["feature_schema_version"] = FEATURE_CONTEXT.get("version") or "0"
    row["feature_config_hash"] = FEATURE_CONTEXT.get("config_hash") or "hash"

    writer = ParquetWriter(str(tmp_path), "daily", src="unit")
    parquet_path = writer.flush_atomic(
        FlushChunk(
            rows=[row],
            min_ts=row["ts_ms"],
            max_ts=row["ts_ms"],
            symbols=[row["symbol"]],
        )
    )

    table = pq.read_table(parquet_path)
    metadata = table.schema.metadata or {}
    schema_version = FEATURE_CONTEXT.get("version")
    schema_hash = FEATURE_CONTEXT.get("schema_hash")

    if schema_version:
        assert metadata.get(b"feature_schema_version") == str(schema_version).encode()
    else:
        assert b"feature_schema_version" not in metadata
    if schema_hash:
        assert metadata.get(b"feature_field_order_hash") == schema_hash.encode()
    order = metadata.get(b"feature_field_order")
    assert order is not None
    assert json.loads(order.decode()) == FEATURE_CONTEXT.get("model_field_order")

    sidecar_path = Path(parquet_path).with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    schema = sidecar["feature_schema"]
    assert schema["field_order_hash"] == FEATURE_CONTEXT.get("schema_hash")
    assert schema["field_order"] == FEATURE_CONTEXT.get("model_field_order")

    FEATURE_CONTEXT["enabled"] = False
    FEATURE_CONTEXT["states"].clear()
    FEATURE_CONTEXT["field_names"] = []
    FEATURE_CONTEXT["model_field_order"] = []
    FEATURE_CONTEXT["schema_hash"] = None
    FEATURE_CONTEXT["order_json"] = None
