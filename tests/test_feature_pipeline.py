from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app.features import FeaturePipeline, split_by_regime


def _make_depth_frame(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    price = np.linspace(10.0, 10.5, num=len(timestamps))
    bid_px = price - 0.01
    ask_px = price + 0.01
    data = {
        "ts": timestamps,
        "price": price,
        "bid_px": bid_px,
        "ask_px": ask_px,
        "bid_qty": np.full(len(timestamps), 5.0),
        "ask_qty": np.full(len(timestamps), 4.0),
    }
    for level in range(1, 11):
        data[f"bid_px_{level}"] = bid_px - 0.0001 * (level - 1)
        data[f"ask_px_{level}"] = ask_px + 0.0001 * (level - 1)
        data[f"bid_qty_{level}"] = np.full(len(timestamps), 5.0 + 0.1 * level)
        data[f"ask_qty_{level}"] = np.full(len(timestamps), 4.0 + 0.1 * level)
    return pd.DataFrame(data)


def test_feature_pipeline_computes_core_features(tmp_path):
    timestamps = pd.date_range("2025-01-01", periods=240, freq="S", tz="UTC")
    frame = _make_depth_frame(timestamps)

    pipeline = FeaturePipeline(
        feature_config=None,
        raw_cache_dir=tmp_path / "raw",
        feature_store_dir=tmp_path / "features",
        progress=False,
    )
    result = pipeline.run(
        frame,
        cache_raw=True,
        materialize=True,
        split="train",
        refresh_stats=True,
    )

    expected_columns = {
        "spread",
        "depth_imbalance",
        "depth_imbalance_l3",
        "queue_bid_ratio",
        "spread_state_1tick",
        "markout_mid_1s",
        "microprice",
        "ofi",
        "skew",
        "kurtosis",
        "regime",
        "vol_bucket",
    }
    assert expected_columns.issubset(result.columns)

    # Normalization stats should be locked in and produce near-zero mean features.
    assert np.isclose(result["spread"].mean(), 0.0, atol=1e-6)
    assert np.isclose(result["depth_imbalance"].mean(), 0.0, atol=1e-6)

    raw_partitions = list((tmp_path / "raw").glob("date=*"))
    assert raw_partitions, "raw parquet cache not materialized"

    metadata_path = tmp_path / "features" / "metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata.get("feature_schema_hash")
    assert metadata.get("normalization", {}).get("strategy") == "standard"

    normalization_path = tmp_path / "features" / "normalization" / "stats.json"
    assert normalization_path.exists()

    metrics_dir = tmp_path / "features" / "metrics"
    assert any(metrics_dir.iterdir()), "monitoring metrics not emitted"


def test_split_by_regime_returns_all_buckets_and_labels():
    timestamps = pd.date_range("2025-01-01", periods=600, freq="S", tz="UTC")
    price = np.sin(np.linspace(0, 10, len(timestamps))) + np.linspace(100, 101, len(timestamps))
    frame = pd.DataFrame({"ts": timestamps, "price": price})
    regimes = split_by_regime(
        frame,
        price_column="price",
        vol_window=60,
        trend_window=120,
        vol_threshold=0.0001,
    )

    assert {"bull", "bear", "range"} == set(regimes.keys())
    total_rows = sum(len(df) for df in regimes.values())
    assert total_rows == len(frame)
    assert all("vol_bucket" in df.columns for df in regimes.values())
