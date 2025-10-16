from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from app.features import FeaturePipeline


def _frame_hash(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=True).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _make_constant_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=120, freq="S", tz="UTC")
    price = np.linspace(50.0, 50.5, num=len(timestamps))
    bid_px = price - 0.01
    ask_px = price + 0.01
    data = {
        "ts": timestamps,
        "price": price,
        "bid_px": bid_px,
        "ask_px": ask_px,
        "bid_qty": np.full(len(timestamps), 8.0),
        "ask_qty": np.full(len(timestamps), 6.0),
    }
    for level in range(1, 6):
        data[f"bid_px_{level}"] = bid_px - 0.00005 * (level - 1)
        data[f"ask_px_{level}"] = ask_px + 0.00005 * (level - 1)
        data[f"bid_qty_{level}"] = np.full(len(timestamps), 8.0 + 0.1 * level)
        data[f"ask_qty_{level}"] = np.full(len(timestamps), 6.0 + 0.1 * level)
    return pd.DataFrame(data)


def test_golden_day_snapshot(tmp_path):
    frame = _make_constant_frame()
    pipeline = FeaturePipeline(
        feature_config=None,
        raw_cache_dir=tmp_path / "raw",
        feature_store_dir=tmp_path / "features",
        progress=False,
    )
    result = pipeline.run(
        frame,
        cache_raw=False,
        materialize=False,
        split="train",
        refresh_stats=True,
    )
    subset = result[["spread", "depth_imbalance"]].round(9)
    expected = pd.DataFrame(0.0, index=subset.index, columns=subset.columns)
    assert _frame_hash(subset) == _frame_hash(expected)


def test_ofi_regression(tmp_path):
    timestamps = pd.date_range("2025-01-01", periods=6, freq="S", tz="UTC")
    bid_px = np.array([100.0, 100.0, 100.1, 100.1, 100.1, 100.0])
    ask_px = np.array([100.02, 100.02, 100.02, 100.05, 100.05, 100.04])
    bid_qty = np.array([5.0, 6.0, 6.5, 6.0, 5.5, 5.0])
    ask_qty = np.array([5.5, 5.5, 5.0, 4.5, 5.0, 5.5])
    frame = pd.DataFrame(
        {
            "ts": timestamps,
            "price": (bid_px + ask_px) / 2.0,
            "bid_px": bid_px,
            "ask_px": ask_px,
            "bid_qty": bid_qty,
            "ask_qty": ask_qty,
        }
    )
    pipeline = FeaturePipeline(
        feature_config=None,
        raw_cache_dir=tmp_path / "raw",
        feature_store_dir=tmp_path / "features",
        progress=False,
        exclude_from_normalization=("ofi",),
    )
    result = pipeline.run(
        frame,
        cache_raw=False,
        materialize=False,
        split="train",
        refresh_stats=True,
    )

    expected_ofi = [0.0]
    for idx in range(1, len(frame)):
        prev_bid_px = bid_px[idx - 1]
        prev_ask_px = ask_px[idx - 1]
        prev_bid_qty = bid_qty[idx - 1]
        prev_ask_qty = ask_qty[idx - 1]
        bid_contrib = (
            bid_qty[idx]
            if bid_px[idx] > prev_bid_px
            else -prev_bid_qty
            if bid_px[idx] < prev_bid_px
            else bid_qty[idx] - prev_bid_qty
        )
        ask_contrib = (
            prev_ask_qty
            if ask_px[idx] < prev_ask_px
            else -ask_qty[idx]
            if ask_px[idx] > prev_ask_px
            else prev_ask_qty - ask_qty[idx]
        )
        expected_ofi.append(bid_contrib - ask_contrib)
    np.testing.assert_allclose(result["ofi"].values, expected_ofi, atol=1e-6)
