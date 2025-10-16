"""Market regime detection and split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict
import pandas as pd


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    RANGE = "range"


@dataclass(slots=True)
class MarketRegimeConfig:
    return_window: int = 60  # minutes
    volatility_window: int = 120  # minutes
    bull_return_threshold: float = 0.01
    bear_return_threshold: float = -0.01
    max_range_volatility: float = 0.005


class MarketRegimeSplitter:
    """Assigns bull/bear/range labels and creates split manifests."""

    def __init__(self, cfg: MarketRegimeConfig) -> None:
        self.cfg = cfg

    def label(self, frame: pd.DataFrame) -> pd.Series:
        closes = frame["price"].astype(float)
        returns = closes.pct_change(self.cfg.return_window).fillna(0.0)
        volatility = (
            closes.pct_change().rolling(self.cfg.volatility_window).std().fillna(0.0)
        )

        labels = pd.Series(index=frame.index, data=MarketRegime.RANGE.value)
        labels = labels.astype("object")

        labels[returns >= self.cfg.bull_return_threshold] = MarketRegime.BULL.value
        labels[returns <= self.cfg.bear_return_threshold] = MarketRegime.BEAR.value
        low_vol_mask = volatility <= self.cfg.max_range_volatility
        labels[(returns.abs() < self.cfg.bull_return_threshold) & low_vol_mask] = MarketRegime.RANGE.value

        return labels.astype("category")

    def split_windows(
        self,
        frame: pd.DataFrame,
        *,
        min_window_minutes: int = 60,
    ) -> Dict[MarketRegime, pd.DataFrame]:
        labels = self.label(frame)
        frame = frame.copy()
        frame["regime"] = labels
        grouped: Dict[MarketRegime, list[pd.DataFrame]] = {
            MarketRegime.BULL: [],
            MarketRegime.BEAR: [],
            MarketRegime.RANGE: [],
        }
        if frame.empty:
            return {regime: pd.DataFrame(columns=frame.columns) for regime in grouped}

        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
        frame = frame.sort_values("ts")
        frame["window_id"] = (frame["ts"].diff().dt.total_seconds().fillna(0).abs() > 60).cumsum()

        for regime_value, subframe in frame.groupby("regime"):
            if subframe.empty:
                continue
            grouped_regime = MarketRegime(regime_value)
            blocks = self._split_into_windows(subframe, min_window_minutes)
            grouped[grouped_regime].extend(blocks)

        return {regime: pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=frame.columns)
                for regime, blocks in grouped.items()}

    def _split_into_windows(self, frame: pd.DataFrame, min_window_minutes: int) -> list[pd.DataFrame]:
        if frame.empty:
            return []
        windows: list[pd.DataFrame] = []
        start_idx = 0
        frame = frame.reset_index(drop=True)
        timestamps = pd.to_datetime(frame["ts"], utc=True)
        for idx in range(1, len(frame)):
            elapsed = (timestamps.iloc[idx] - timestamps.iloc[start_idx]).total_seconds() / 60.0
            if elapsed >= min_window_minutes:
                windows.append(frame.iloc[start_idx:idx].copy())
                start_idx = idx
        if start_idx < len(frame) - 1:
            windows.append(frame.iloc[start_idx:].copy())
        return windows
