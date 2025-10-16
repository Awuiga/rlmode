from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st

from app.features import FeaturePipeline


def _build_frame(increments: list[int]) -> pd.DataFrame:
    base_time = pd.Timestamp("2025-01-01", tz="UTC")
    current = base_time
    timestamps = []
    for delta in increments:
        current = current + pd.Timedelta(seconds=int(delta))
        timestamps.append(current)
    price = np.linspace(100.0, 101.0, num=len(timestamps))
    bid_px = price - 0.01
    ask_px = price + 0.01
    data = {
        "ts": timestamps,
        "price": price,
        "bid_px": bid_px,
        "ask_px": ask_px,
        "bid_qty": np.full(len(timestamps), 10.0),
        "ask_qty": np.full(len(timestamps), 9.0),
    }
    for level in range(1, 6):
        data[f"bid_px_{level}"] = bid_px - 0.0001 * (level - 1)
        data[f"ask_px_{level}"] = ask_px + 0.0001 * (level - 1)
        data[f"bid_qty_{level}"] = np.full(len(timestamps), 10.0 + 0.2 * level)
        data[f"ask_qty_{level}"] = np.full(len(timestamps), 9.0 + 0.2 * level)
    return pd.DataFrame(data)


@settings(max_examples=15, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=3), min_size=20, max_size=40))
def test_pipeline_handles_timestamp_anomalies(tmp_path, increments: list[int]) -> None:
    if all(delta == 0 for delta in increments):
        increments[0] = 1
    frame = _build_frame(increments)
    features_dir = tmp_path / f"features_{uuid.uuid4().hex}"
    raw_dir = tmp_path / f"raw_{uuid.uuid4().hex}"
    pipeline = FeaturePipeline(
        feature_config=None,
        raw_cache_dir=raw_dir,
        feature_store_dir=features_dir,
        progress=False,
    )
    try:
        result = pipeline.run(
            frame,
            cache_raw=False,
            materialize=False,
            split="train",
            refresh_stats=True,
        )
    except RuntimeError as exc:
        assert "schema" in str(exc).lower() or "timestamp" in str(exc).lower()
    else:
        assert len(result) == len(frame)
