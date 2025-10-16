"""Dataset materialisation and split helpers for RL scalper."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .regime import MarketRegime, MarketRegimeConfig, MarketRegimeSplitter


@dataclass(slots=True)
class DatasetSplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    micro_window: int = 60
    macro_window: int = 60
    micro_columns: Sequence[str] = field(
        default_factory=lambda: [
            "price",
            "qty",
            "spread",
            "depth_imbalance",
            "order_imbalance",
            "realized_volatility",
        ]
    )
    macro_columns: Sequence[str] = field(
        default_factory=lambda: [
            "price",
            "funding_rate",
            "cumulative_volume",
        ]
    )
    meta_columns: Sequence[str] = field(
        default_factory=lambda: [
            "price",
            "spread",
            "depth_bids",
            "depth_asks",
            "depth_imbalance",
            "funding_rate",
            "order_imbalance",
            "realized_volatility",
        ]
    )

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError("train + val + test ratios must sum to 1.0")


class RLScalperDataset:
    """Prepare model-ready tensors (micro/macro splits) from collected parquet files."""

    def __init__(
        self,
        *,
        parquet_paths: Sequence[Path],
        output_root: Path,
        split_cfg: DatasetSplitConfig,
        regime_cfg: MarketRegimeConfig,
    ) -> None:
        self.parquet_paths = parquet_paths
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.split_cfg = split_cfg
        self.regime_splitter = MarketRegimeSplitter(regime_cfg)

    def build(self) -> Dict[str, Path]:
        frame = self._load_and_concat()
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
        frame = frame.sort_values("ts").reset_index(drop=True)
        frame["regime"] = self.regime_splitter.label(frame)
        splits = self._split_by_regime(frame)

        artifacts: Dict[str, Path] = {}
        for split_name, split_frame in splits.items():
            if split_frame.empty:
                continue
            tensors = self._materialise(split_frame)
            path = self.output_root / f"{split_name}.npz"
            np.savez(path, **tensors)
            artifacts[split_name] = path
        return artifacts

    def _load_and_concat(self) -> pd.DataFrame:
        frames = [pd.read_parquet(path) for path in self.parquet_paths]
        if not frames:
            raise ValueError("no parquet paths provided")
        return pd.concat(frames, ignore_index=True)

    def _split_by_regime(self, frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        splits: Dict[str, List[pd.DataFrame]] = {"train": [], "val": [], "test": []}
        ratios = (self.split_cfg.train_ratio, self.split_cfg.val_ratio, self.split_cfg.test_ratio)
        for regime in MarketRegime:
            regime_frame = frame[frame["regime"] == regime.value]
            if regime_frame.empty:
                continue
            n = len(regime_frame)
            train_end = int(n * ratios[0])
            val_end = train_end + int(n * ratios[1])
            splits["train"].append(regime_frame.iloc[:train_end])
            splits["val"].append(regime_frame.iloc[train_end:val_end])
            splits["test"].append(regime_frame.iloc[val_end:])

        return {
            split: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=frame.columns)
            for split, frames in splits.items()
        }

    def _materialise(self, frame: pd.DataFrame) -> Dict[str, np.ndarray]:
        micro = self._build_micro_sequences(frame)
        macro = self._build_macro_vectors(frame)
        start_idx = self.split_cfg.micro_window - 1
        timestamps = frame["ts"].iloc[start_idx:].to_numpy()
        meta = frame[self.split_cfg.meta_columns].iloc[start_idx:].to_numpy(dtype=np.float32)
        return {
            "micro": micro,
            "macro": macro,
            "meta": meta,
            "ts": timestamps.astype("datetime64[ns]"),
        }

    def _build_micro_sequences(self, frame: pd.DataFrame) -> np.ndarray:
        cols = list(self.split_cfg.micro_columns)
        window = self.split_cfg.micro_window
        matrix = frame[cols].to_numpy(dtype=np.float32)
        if matrix.shape[0] < window:
            raise ValueError(f"not enough rows ({matrix.shape[0]}) to build micro window {window}")
        seqs = [
            matrix[idx - window : idx]
            for idx in range(window, matrix.shape[0] + 1)
        ]
        return np.stack(seqs, axis=0).astype(np.float32)

    def _build_macro_vectors(self, frame: pd.DataFrame) -> np.ndarray:
        cols = list(self.split_cfg.macro_columns)
        window = self.split_cfg.macro_window
        df = (
            frame[cols]
            .rolling(window=window, min_periods=window)
            .agg(["mean", "std", "min", "max"])
            .dropna()
        )
        flattened = df.values.astype(np.float32)
        return flattened
