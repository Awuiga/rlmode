from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class BarrierSpec:
    """Triple-barrier configuration."""

    tp_pct: float
    sl_pct: float
    max_horizon: pd.Timedelta | int

    def horizon_ms(self) -> int:
        if isinstance(self.max_horizon, pd.Timedelta):
            return int(self.max_horizon.total_seconds() * 1000)
        return int(self.max_horizon)


def _as_direction(series: pd.Series | None) -> np.ndarray:
    if series is None:
        return np.ones(0, dtype=np.int8)
    values = series.fillna(1).to_numpy()
    dirs = np.ones_like(values, dtype=np.int8)
    for idx, val in enumerate(values):
        if isinstance(val, (int, float)):
            dirs[idx] = 1 if val >= 0 else -1
        else:
            sval = str(val).upper()
            if sval in {"SELL", "S", "-1"}:
                dirs[idx] = -1
            else:
                dirs[idx] = 1
    return dirs


def triple_barrier(
    events: pd.DataFrame,
    *,
    price_col: str,
    barrier: BarrierSpec,
    time_col: str = "ts_ms",
    side_col: Optional[str] = None,
) -> pd.DataFrame:
    """Apply triple barrier labeling to a time-ordered event dataframe.

    Returns a dataframe aligned to the input index with columns:
      - label: 'tp' | 'sl' | 'timeout'
      - label_int: 1 for take-profit, 0 otherwise
      - outcome_ret: realised return scaled by trade direction
      - event_end_ts: timestamp where barrier triggered or timeout
      - holding_ms: duration in milliseconds until barrier exit
    """

    if price_col not in events.columns:
        raise KeyError(f"price column '{price_col}' not found")
    if time_col not in events.columns:
        raise KeyError(f"time column '{time_col}' not found")

    prices = events[price_col].astype(float).to_numpy()
    times = events[time_col].astype(np.int64).to_numpy()
    dirs = _as_direction(events[side_col] if side_col else None)
    if dirs.size == 0:
        dirs = np.ones_like(prices, dtype=np.int8)
    horizon_ms = max(1, barrier.horizon_ms())

    labels: List[str] = []
    label_ints: List[int] = []
    returns: List[float] = []
    end_times: List[int] = []
    holdings: List[int] = []

    n = len(events)
    for idx in range(n):
        entry_price = prices[idx]
        if not np.isfinite(entry_price) or entry_price <= 0:
            labels.append("nan")
            label_ints.append(0)
            returns.append(np.nan)
            end_times.append(int(times[idx]))
            holdings.append(0)
            continue

        direction = dirs[idx] if idx < len(dirs) else 1
        upper = barrier.tp_pct
        lower = -barrier.sl_pct
        expiry_ts = times[idx] + horizon_ms

        outcome = "timeout"
        outcome_ret = 0.0
        exit_ts = expiry_ts

        j = idx + 1
        while j < n and times[j] <= expiry_ts:
            future_ret = direction * ((prices[j] - entry_price) / entry_price)
            if future_ret >= upper:
                outcome = "tp"
                outcome_ret = future_ret
                exit_ts = int(times[j])
                break
            if future_ret <= lower:
                outcome = "sl"
                outcome_ret = future_ret
                exit_ts = int(times[j])
                break
            outcome_ret = future_ret
            exit_ts = int(times[j])
            j += 1

        if outcome == "timeout":
            # Ensure we use the last available point within horizon or the last observation
            if j >= n or times[j] > expiry_ts:
                # already set to last iter value; if no iteration happened keep zero
                if j == idx + 1:
                    # no future samples; use current price
                    outcome_ret = 0.0
                    exit_ts = int(expiry_ts)
            else:
                outcome_ret = direction * ((prices[j] - entry_price) / entry_price)
                exit_ts = int(times[j])

        labels.append(outcome)
        label_ints.append(1 if outcome == "tp" else 0)
        returns.append(outcome_ret)
        holdings.append(max(0, exit_ts - int(times[idx])))
        end_times.append(exit_ts)

    result = pd.DataFrame(
        {
            "label": labels,
            "label_int": label_ints,
            "outcome_ret": returns,
            "event_end_ts": end_times,
            "holding_ms": holdings,
        },
        index=events.index,
    )
    return result


@dataclass
class PrimarySignalConfig:
    ofi_col: str = "ofi"
    qi_col: str = "qi"
    ofi_threshold: float = 5.0
    qi_threshold: float = 0.1
    spread_regime_col: Optional[str] = "spread_regime"
    max_spread_regime: Optional[float] = 1.0
    direction_col: str = "side"
    price_col: str = "microprice"
    time_col: str = "ts_ms"


def generate_primary_entries(df: pd.DataFrame, cfg: PrimarySignalConfig) -> pd.DataFrame:
    """Create coarse entry candidates based on OFI/QI imbalance rules."""
    if cfg.ofi_col not in df.columns or cfg.qi_col not in df.columns:
        raise KeyError("OFI/QI columns not present in dataframe")

    cond = (df[cfg.ofi_col].abs() >= cfg.ofi_threshold) & (df[cfg.qi_col].abs() >= cfg.qi_threshold)
    if cfg.spread_regime_col and cfg.spread_regime_col in df.columns and cfg.max_spread_regime is not None:
        cond &= df[cfg.spread_regime_col] <= cfg.max_spread_regime

    entries = df.loc[cond].copy()
    if entries.empty:
        return entries

    direction = np.where(entries[cfg.ofi_col] >= 0, 1, -1)
    entries[cfg.direction_col] = direction
    if cfg.price_col in df.columns:
        entries[cfg.price_col] = entries[cfg.price_col].astype(float)
    if cfg.time_col in df.columns:
        entries[cfg.time_col] = entries[cfg.time_col].astype(np.int64)
    return entries


def apply_meta_label(entries: pd.DataFrame, barriers: pd.DataFrame) -> pd.DataFrame:
    """Join triple-barrier outcomes to primary entries and derive meta-label."""
    merged = entries.join(barriers, how="inner")
    merged["meta_label"] = (merged["label"] == "tp").astype(int)
    return merged


def build_meta_dataset(
    df: pd.DataFrame,
    *,
    primary_cfg: PrimarySignalConfig,
    barrier: BarrierSpec,
    feature_cols: Sequence[str],
    dropna: bool = True,
) -> pd.DataFrame:
    """Pipeline helper: primary entries -> triple barrier -> meta dataset."""
    entries = generate_primary_entries(df, primary_cfg)
    if entries.empty:
        return entries
    barriers = triple_barrier(entries, price_col=primary_cfg.price_col, barrier=barrier, time_col=primary_cfg.time_col, side_col=primary_cfg.direction_col)
    dataset = apply_meta_label(entries, barriers)
    cols = list(feature_cols) + ["meta_label", "label", "label_int", "outcome_ret", "holding_ms"]
    dataset = dataset[cols].copy()
    if dropna:
        dataset = dataset.dropna(subset=feature_cols)
    return dataset
