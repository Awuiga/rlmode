from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from .core import compute_features_offline, make_feature_state

__all__ = [
    "FeaturePipeline",
    "FeatureTransformation",
    "NormalizationAccumulator",
    "split_by_regime",
]

DEFAULT_DEPTH_LEVELS = (1, 3, 5, 10)
DEFAULT_MARKOUT_HORIZONS = (1, 3, 5)
DEFAULT_VOL_BUCKETS = (0.0005, 0.0015)
SCHEMA_VERSION = "4"


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class FeatureTransformation:
    """Transformation metadata for the modular feature pipeline."""

    name: str
    func: Callable[[pd.DataFrame, "FeaturePipeline"], pd.Series | pd.DataFrame]
    required_columns: Sequence[str] = field(default_factory=tuple)
    description: str | None = None


@dataclass
class NormalizationAccumulator:
    """Online aggregator for normalization statistics."""

    counts: Dict[str, int] = field(default_factory=dict)
    means: Dict[str, float] = field(default_factory=dict)
    m2: Dict[str, float] = field(default_factory=dict)
    strategy: str = "standard"
    version: str = "1"

    def update(self, frame: pd.DataFrame, columns: Sequence[str]) -> None:
        for col in columns:
            series = frame[col].dropna().astype(float)
            if series.empty:
                continue
            count_prev = self.counts.get(col, 0)
            mean_prev = self.means.get(col, 0.0)
            m2_prev = self.m2.get(col, 0.0)

            batch_count = int(series.count())
            batch_mean = float(series.mean())
            deviations = series.to_numpy(dtype=float) - batch_mean
            batch_m2 = float(np.dot(deviations, deviations))

            total_count = count_prev + batch_count
            if total_count == 0:
                continue

            delta = batch_mean - mean_prev
            mean_new = mean_prev + delta * batch_count / total_count
            m2_new = m2_prev + batch_m2 + delta * delta * count_prev * batch_count / total_count

            self.counts[col] = total_count
            self.means[col] = mean_new
            self.m2[col] = m2_new

    def statistics(self) -> Dict[str, Tuple[float, float]]:
        stats: Dict[str, Tuple[float, float]] = {}
        for col, count in self.counts.items():
            mean = self.means.get(col, 0.0)
            if count <= 1:
                stats[col] = (mean, 1.0)
                continue
            variance = self.m2.get(col, 0.0) / (count - 1)
            std = float(np.sqrt(max(variance, 1e-12)))
            stats[col] = (mean, std)
        return stats

    def to_json(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "strategy": self.strategy,
            "counts": self.counts,
            "means": self.means,
            "m2": self.m2,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "NormalizationAccumulator":
        return cls(
            counts={k: int(v) for k, v in dict(payload.get("counts") or {}).items()},
            means={k: float(v) for k, v in dict(payload.get("means") or {}).items()},
            m2={k: float(v) for k, v in dict(payload.get("m2") or {}).items()},
            strategy=str(payload.get("strategy") or "standard"),
            version=str(payload.get("version") or "1"),
        )


class FeaturePipeline:
    """Modular feature engineering pipeline with QC, caching, normalization, and monitoring."""

    def __init__(
        self,
        *,
        feature_config: Optional[Mapping[str, object]] = None,
        depth_top_k: int = 5,
        transforms: Optional[Sequence[FeatureTransformation]] = None,
        tz: timezone = timezone.utc,
        ts_column: str = "ts",
        price_column: str = "price",
        bid_price_columns: Sequence[str] = ("bid_px", "bid1", "best_bid"),
        ask_price_columns: Sequence[str] = ("ask_px", "ask1", "best_ask"),
        bid_size_columns: Sequence[str] = ("bid_qty", "bid_sz", "depth_bids", "bid_qty_1"),
        ask_size_columns: Sequence[str] = ("ask_qty", "ask_sz", "depth_asks", "ask_qty_1"),
        raw_cache_dir: Optional[Path] = None,
        feature_store_dir: Optional[Path] = None,
        materialization_freq: str = "D",
        realized_vol_window: int = 60,
        shape_window: int = 120,
        anomalies_zscore: float = 6.0,
        gap_threshold_ms: int = 1_000,
        sync_tolerance_ms: int = 1_500,
        progress: bool = True,
        stateful_batch_size: int = 5_000,
        normalization_dir: Optional[Path] = None,
        normalization_strategy: str = "standard",
        exclude_from_normalization: Optional[Sequence[str]] = None,
        diagnostics_prefixes: Sequence[str] = ("markout_",),
        monitoring_metrics_dir: Optional[Path] = None,
        pipeline_version: str = "2.0",
        depth_levels: Sequence[int] = DEFAULT_DEPTH_LEVELS,
        markout_horizons: Sequence[int] = DEFAULT_MARKOUT_HORIZONS,
        volatility_bins: Sequence[float] = DEFAULT_VOL_BUCKETS,
        regime_vol_window: int = 300,
        regime_trend_window: int = 900,
        regime_vol_threshold: float = 0.002,
    ) -> None:
        self.depth_top_k = depth_top_k
        self.price_column = price_column
        self.ts_column = ts_column
        self.bid_price_columns = tuple(bid_price_columns)
        self.ask_price_columns = tuple(ask_price_columns)
        self.bid_size_columns = tuple(bid_size_columns)
        self.ask_size_columns = tuple(ask_size_columns)
        self.tz = tz
        self.realized_vol_window = realized_vol_window
        self.shape_window = shape_window
        self.anomalies_zscore = anomalies_zscore
        self.gap_threshold_ms = gap_threshold_ms
        self.sync_tolerance_ms = sync_tolerance_ms
        self.progress = progress
        self.stateful_batch_size = max(1, stateful_batch_size)
        self.raw_cache_dir = Path(raw_cache_dir) if raw_cache_dir else None
        self.feature_store_dir = Path(feature_store_dir) if feature_store_dir else None
        self.materialization_freq = materialization_freq
        self.feature_config = feature_config
        self.feature_state = (
            make_feature_state(feature_config) if feature_config is not None else None
        )
        self._state_per_symbol: MutableMapping[str, object] = {}
        self.transforms: List[FeatureTransformation] = list(transforms or self._default_transforms())
        self.normalization_strategy = normalization_strategy
        self.exclude_from_normalization = set(exclude_from_normalization or ())
        self.diagnostics_prefixes = tuple(diagnostics_prefixes)
        self.pipeline_version = pipeline_version
        self.depth_levels = tuple(sorted(set(depth_levels)))
        self.markout_horizons = tuple(sorted(set(markout_horizons)))
        self.volatility_bins = tuple(sorted(volatility_bins))
        self.regime_vol_window = regime_vol_window
        self.regime_trend_window = regime_trend_window
        self.regime_vol_threshold = regime_vol_threshold
        self._temp_cache: Dict[str, object] = {}

        if normalization_dir:
            self.normalization_dir = Path(normalization_dir)
        elif self.feature_store_dir:
            self.normalization_dir = self.feature_store_dir / "normalization"
        else:
            self.normalization_dir = Path("normalization")
        self.normalization_dir.mkdir(parents=True, exist_ok=True)
        self._normalization_path = self.normalization_dir / "stats.json"
        self._normalization_accumulator: NormalizationAccumulator | None = None

        if monitoring_metrics_dir:
            self.monitoring_dir = Path(monitoring_metrics_dir)
        elif self.feature_store_dir:
            self.monitoring_dir = self.feature_store_dir / "metrics"
        else:
            self.monitoring_dir = Path("metrics")
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)

        if self.feature_store_dir:
            self.feature_store_dir.mkdir(parents=True, exist_ok=True)
        if self.raw_cache_dir:
            self.raw_cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        frame: pd.DataFrame,
        *,
        symbol: Optional[str] = None,
        cache_raw: bool = True,
        materialize: bool = True,
        split: str = "train",
        refresh_stats: bool = False,
    ) -> pd.DataFrame:
        """Execute the pipeline and return a feature-enriched dataframe."""
        if frame.empty:
            logger.warning("FeaturePipeline received empty dataframe; skipping processing.")
            return frame.copy()

        if split not in {"train", "val", "test", "infer"}:
            raise ValueError(f"Unsupported split '{split}'")

        df = frame.copy()
        df.sort_values(self.ts_column, inplace=True)
        df.reset_index(drop=True, inplace=True)

        ts_series = self._ensure_datetime(df[self.ts_column])
        df[self.ts_column] = ts_series

        quality_metrics = self._log_data_quality(df)
        sync_metrics = self._verify_l2_trade_sync(df)
        df = self._clean_anomalies(df)

        self._temp_cache = {}
        stateful = self._compute_stateful_features(df, symbol=symbol)
        df = pd.concat([df.reset_index(drop=True), stateful.reset_index(drop=True)], axis=1)

        df["mid"] = self._mid_price_series(df)
        self._temp_cache["returns"] = df[self.price_column].astype(float).pct_change().fillna(0.0)
        self._temp_cache["mid_returns"] = df["mid"].pct_change().fillna(0.0)

        additional_frames: List[pd.DataFrame] = []
        iterator: Iterable[FeatureTransformation] = self.transforms
        for transform in tqdm(iterator, disable=not self.progress, desc="Feature transforms"):
            missing = [col for col in transform.required_columns if col not in df.columns]
            if missing:
                logger.warning(
                    "Skipping feature '{}' due to missing columns: {}", transform.name, missing
                )
                continue
            try:
                result = transform.func(df, self)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Failed to compute feature '%s': %s", transform.name, exc)
                continue
            if isinstance(result, pd.DataFrame):
                additional_frames.append(result)
            else:
                df[transform.name] = result
        if additional_frames:
            df = pd.concat([df] + additional_frames, axis=1)

        df = self._annotate_regimes(df)
        df = self._apply_normalization(df, split=split, refresh=refresh_stats)

        metrics = self._collect_monitoring_metrics(df, quality_metrics, sync_metrics)
        self._log_summary(df, metrics)
        self._export_metrics(df, metrics)

        schema_signature = self._compute_schema_signature(df)
        schema_hash = _sha256(json.dumps(schema_signature, sort_keys=True))
        self._ensure_schema_compatibility(schema_signature, schema_hash)

        if cache_raw and self.raw_cache_dir is not None:
            self._cache_raw_data(frame, ts_series)
        if materialize and self.feature_store_dir is not None:
            self._materialize_features(df, schema_signature, schema_hash)
        return df

    # ------------------------------------------------------------------
    # Transform definitions
    # ------------------------------------------------------------------

    def _default_transforms(self) -> Sequence[FeatureTransformation]:
        return (
            FeatureTransformation(
                name="spread",
                func=lambda df, pipeline: pipeline._resolve_series(
                    df, pipeline.ask_price_columns
                )
                - pipeline._resolve_series(df, pipeline.bid_price_columns),
                required_columns=list(set(self.ask_price_columns + self.bid_price_columns)),
                description="Best-ask minus best-bid spread.",
            ),
            FeatureTransformation(
                name="depth_imbalance",
                func=lambda df, pipeline: pipeline._calc_depth_imbalance(df),
                required_columns=list(set(self.bid_size_columns + self.ask_size_columns)),
                description="Top-of-book depth imbalance.",
            ),
            FeatureTransformation(
                name="depth_imbalance_multi",
                func=lambda df, pipeline: pipeline._calc_multi_depth_imbalance(df),
                description="Depth imbalances across multiple levels.",
            ),
            FeatureTransformation(
                name="queue_proxies",
                func=lambda df, pipeline: pipeline._calc_queue_proxies(df),
                description="Queue position proxies at touch.",
            ),
            FeatureTransformation(
                name="spread_state",
                func=lambda df, pipeline: pipeline._calc_spread_state(df),
                description="One-hot spread regime indicators.",
            ),
            FeatureTransformation(
                name="markout_mid",
                func=lambda df, pipeline: pipeline._calc_markout_labels(df),
                required_columns=(self.ts_column,),
                description="Diagnostic markout labels from mid price.",
            ),
            FeatureTransformation(
                name="realized_volatility",
                func=lambda df, pipeline: pipeline._calc_realized_vol(df),
                required_columns=(self.price_column,),
                description=f"Rolling {self.realized_vol_window}s volatility of returns.",
            ),
            FeatureTransformation(
                name="microprice",
                func=lambda df, pipeline: pipeline._calc_microprice(df),
                required_columns=list(
                    set(
                        self.bid_price_columns
                        + self.ask_price_columns
                        + self.bid_size_columns
                        + self.ask_size_columns
                    )
                ),
                description="Microprice computed from best quotes and sizes.",
            ),
            FeatureTransformation(
                name="ofi",
                func=lambda df, pipeline: pipeline._calc_ofi(df),
                required_columns=list(
                    set(
                        self.bid_price_columns
                        + self.ask_price_columns
                        + self.bid_size_columns
                        + self.ask_size_columns
                    )
                ),
                description="Order flow imbalance at top of book.",
            ),
            FeatureTransformation(
                name="shape_stats",
                func=lambda df, pipeline: pipeline._calc_shape_stats(df),
                required_columns=(self.price_column,),
                description=f"Rolling {self.shape_window}s skewness and kurtosis of returns.",
            ),
        )

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def _compute_stateful_features(
        self,
        df: pd.DataFrame,
        *,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        if self.feature_state is None or self.feature_config is None:
            return pd.DataFrame(index=df.index)

        rows = df.to_dict(orient="records")
        features: List[Dict[str, float]] = []
        state = self._state_for(symbol)
        for start in range(0, len(rows), self.stateful_batch_size):
            batch = rows[start : start + self.stateful_batch_size]
            for record in batch:
                numeric_record = self._prepare_market_record(record)
                values = compute_features_offline(
                    numeric_record,
                    state=state,
                    depth_top_k=self.depth_top_k,
                )
                features.append(values)
        if not features:
            return pd.DataFrame(index=df.index)
        feature_frame = pd.DataFrame.from_records(features).fillna(0.0)
        return feature_frame

    def _state_for(self, symbol: Optional[str]) -> object:
        if symbol is None or not symbol:
            return self.feature_state
        state = self._state_per_symbol.get(symbol)
        if state is None:
            state = make_feature_state(self.feature_config)
            self._state_per_symbol[symbol] = state
        return state

    # ------------------------------------------------------------------
    # Data quality checks
    # ------------------------------------------------------------------

    def _log_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        ts = df[self.ts_column]
        if ts.is_monotonic_increasing is False:
            backward = (ts.diff() < pd.Timedelta(0)).sum()
            metrics["lag_count"] = float(backward)
            logger.warning("Detected {} timestamp regressions (lags) in feed.", int(backward))
        else:
            metrics["lag_count"] = 0.0
        diffs = ts.diff().dropna()
        gaps = diffs[diffs > pd.Timedelta(milliseconds=self.gap_threshold_ms)]
        metrics["gap_count"] = float(len(gaps))
        if not gaps.empty:
            metrics["max_gap_seconds"] = float(gaps.max().total_seconds())
            logger.warning(
                "Detected {} timestamp gaps above {}ms (max gap={}ms).",
                len(gaps),
                self.gap_threshold_ms,
                int(gaps.max().total_seconds() * 1000),
            )
        else:
            metrics["max_gap_seconds"] = 0.0
        missing_cols = df.isna().sum()
        missing_total = int(missing_cols.sum())
        metrics["missing_total"] = float(missing_total)
        if missing_total:
            missing_report = {
                col: int(count) for col, count in missing_cols.items() if int(count) > 0
            }
            metrics["missing_cols"] = len(missing_report)
            logger.warning("Missing values detected: {}", missing_report)
        else:
            metrics["missing_cols"] = 0.0
        return metrics

    def _verify_l2_trade_sync(self, df: pd.DataFrame) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        trade_ts_cols = [col for col in ("trade_ts", "last_trade_ts") if col in df.columns]
        ts = df[self.ts_column]
        if not trade_ts_cols:
            price = df.get(self.price_column)
            mid = df.get("mid")
            if price is None or mid is None:
                logger.info("L2/trade sync check skipped (no trade timestamp or mid price columns).")
                metrics["sync_outliers"] = 0.0
                metrics["sync_max_shift"] = 0.0
                return metrics
            corr = price.pct_change().corr(mid.pct_change())
            metrics["price_mid_corr"] = float(corr) if corr is not None else 0.0
            if corr is not None and corr < 0.2:
                logger.warning(
                    "Low correlation between trade price and mid moves (corr={:.3f}); possible misalignment.",
                    corr,
                )
            metrics["sync_outliers"] = 0.0
            metrics["sync_max_shift"] = 0.0
            return metrics
        tolerance = pd.Timedelta(milliseconds=self.sync_tolerance_ms)
        total_outliers = 0
        max_shift = 0.0
        for col in trade_ts_cols:
            trade_ts = self._ensure_datetime(df[col])
            delta = (trade_ts - ts).abs()
            outliers = delta[delta > tolerance]
            total_outliers += int(len(outliers))
            if not outliers.empty:
                max_shift = max(max_shift, float(outliers.max().total_seconds()))
        metrics["sync_outliers"] = float(total_outliers)
        metrics["sync_max_shift"] = max_shift
        return metrics

    def _clean_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "spread" in df.columns:
            spread = df["spread"].astype(float)
        else:
            spread = self._resolve_series(df, self.ask_price_columns) - self._resolve_series(
                df, self.bid_price_columns
            )
            df["spread"] = spread

        depth_cols = [
            col for col in (*self.bid_size_columns, *self.ask_size_columns) if col in df.columns
        ]
        metrics = {"spread": spread}
        for col in depth_cols:
            metrics[col] = df[col].astype(float)
        zscore_limit = self.anomalies_zscore
        cleaned_rows = 0
        for name, series in metrics.items():
            median = series.median()
            mad = (series - median).abs().median()
            if mad == 0 or np.isnan(mad):
                continue
            zscore = (series - median) / (1.4826 * mad)
            mask = zscore.abs() > zscore_limit
            if mask.any():
                cleaned_rows += int(mask.sum())
                df.loc[mask, name] = median
        if cleaned_rows:
            logger.info("Clipped {} anomalous observations based on MAD threshold.", cleaned_rows)
        return df

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    def _calc_depth_imbalance(self, df: pd.DataFrame) -> pd.Series:
        bid = self._resolve_series(df, self.bid_size_columns)
        ask = self._resolve_series(df, self.ask_size_columns)
        denom = bid + ask
        imbalance = np.divide(
            bid - ask,
            denom,
            out=np.zeros_like(denom, dtype=float),
            where=denom != 0,
        )
        return pd.Series(imbalance, index=df.index)

    def _calc_multi_depth_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        bid_levels = self._collect_depth_levels(df, side="bid")
        ask_levels = self._collect_depth_levels(df, side="ask")
        results: Dict[str, pd.Series] = {}
        for depth in self.depth_levels:
            bid_sum = bid_levels[bid_levels.columns[bid_levels.columns <= depth]].sum(axis=1)
            ask_sum = ask_levels[ask_levels.columns[ask_levels.columns <= depth]].sum(axis=1)
            denom = bid_sum + ask_sum
            values = np.divide(
                bid_sum - ask_sum,
                denom,
                out=np.zeros_like(denom, dtype=float),
                where=denom != 0,
            )
            results[f"depth_imbalance_l{depth}"] = pd.Series(values, index=df.index)
        return pd.DataFrame(results)

    def _calc_queue_proxies(self, df: pd.DataFrame) -> pd.DataFrame:
        bid_levels = self._collect_depth_levels(df, side="bid")
        ask_levels = self._collect_depth_levels(df, side="ask")
        lvl1_bid = bid_levels.get(1, pd.Series(0.0, index=df.index)).astype(float)
        lvl1_ask = ask_levels.get(1, pd.Series(0.0, index=df.index)).astype(float)
        total_bid = bid_levels.sum(axis=1).replace(0.0, np.nan)
        total_ask = ask_levels.sum(axis=1).replace(0.0, np.nan)
        bid_ratio = (lvl1_bid / total_bid).fillna(0.0)
        ask_ratio = (lvl1_ask / total_ask).fillna(0.0)
        bid_flow = lvl1_bid.diff().fillna(0.0)
        ask_flow = lvl1_ask.diff().fillna(0.0)
        return pd.DataFrame(
            {
                "queue_bid_ratio": bid_ratio,
                "queue_ask_ratio": ask_ratio,
                "queue_bid_touch_flow": bid_flow,
                "queue_ask_touch_flow": ask_flow,
            }
        )

    def _calc_spread_state(self, df: pd.DataFrame) -> pd.DataFrame:
        spread = df.get("spread")
        if spread is None:
            spread = self._resolve_series(df, self.ask_price_columns) - self._resolve_series(
                df, self.bid_price_columns
            )
        positive = spread.replace(0, np.nan).dropna()
        tick = float(positive.min()) if not positive.empty else 0.0
        if tick == 0.0:
            tick = float(spread.replace(0, np.nan).median() or 1.0)
        one_tick = (spread - tick).abs() <= tick * 0.1
        gt_one = spread > tick * 1.5
        state = pd.DataFrame(
            {
                "spread_state_1tick": one_tick.astype(int),
                "spread_state_gt1tick": gt_one.astype(int),
            }
        )
        state["spread_state_zero"] = (spread <= 0).astype(int)
        return state

    def _calc_markout_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = self._ensure_datetime(df[self.ts_column])
        mid = self._mid_price_series(df)
        base = pd.DataFrame({"ts": ts, "mid": mid}).sort_values("ts")
        targets = base.rename(columns={"ts": "ts_future", "mid": "mid_future"})
        result: Dict[str, pd.Series] = {}
        for horizon in self.markout_horizons:
            lookup = base.copy()
            lookup["lookup_ts"] = lookup["ts"] + pd.to_timedelta(horizon, unit="s")
            merged = pd.merge_asof(
                lookup.sort_values("lookup_ts"),
                targets,
                left_on="lookup_ts",
                right_on="ts_future",
                direction="forward",
                tolerance=pd.Timedelta(seconds=horizon * 5),
            )
            delta = merged["mid_future"] - lookup["mid"]
            result[f"markout_mid_{horizon}s"] = delta.reindex(base.index).fillna(0.0)
        return pd.DataFrame(result).reindex(df.index).fillna(0.0)

    def _calc_realized_vol(self, df: pd.DataFrame) -> pd.Series:
        returns = self._temp_cache.get("returns")
        if returns is None:
            returns = df[self.price_column].astype(float).pct_change().fillna(0.0)
            self._temp_cache["returns"] = returns
        window = max(5, self.realized_vol_window)
        vol = returns.rolling(window, min_periods=window // 3).std().fillna(0.0)
        return vol

    def _calc_microprice(self, df: pd.DataFrame) -> pd.Series:
        bid_px = self._resolve_series(df, self.bid_price_columns)
        ask_px = self._resolve_series(df, self.ask_price_columns)
        bid_sz = self._resolve_series(df, self.bid_size_columns)
        ask_sz = self._resolve_series(df, self.ask_size_columns)
        total = bid_sz + ask_sz
        micro = np.divide(
            ask_px * bid_sz + bid_px * ask_sz,
            total,
            out=np.zeros_like(total, dtype=float),
            where=total != 0,
        )
        return pd.Series(micro, index=df.index)

    def _calc_ofi(self, df: pd.DataFrame) -> pd.Series:
        bid_px = self._resolve_series(df, self.bid_price_columns)
        ask_px = self._resolve_series(df, self.ask_price_columns)
        bid_sz = self._resolve_series(df, self.bid_size_columns)
        ask_sz = self._resolve_series(df, self.ask_size_columns)

        prev_bid_px = bid_px.shift(1).fillna(bid_px.iloc[0])
        prev_bid_sz = bid_sz.shift(1).fillna(bid_sz.iloc[0])
        prev_ask_px = ask_px.shift(1).fillna(ask_px.iloc[0])
        prev_ask_sz = ask_sz.shift(1).fillna(ask_sz.iloc[0])

        bid_contrib = np.where(
            bid_px > prev_bid_px,
            bid_sz,
            np.where(
                bid_px < prev_bid_px,
                -prev_bid_sz,
                bid_sz - prev_bid_sz,
            ),
        )
        ask_contrib = np.where(
            ask_px < prev_ask_px,
            prev_ask_sz,
            np.where(
                ask_px > prev_ask_px,
                -ask_sz,
                prev_ask_sz - ask_sz,
            ),
        )
        ofi = bid_contrib - ask_contrib
        return pd.Series(ofi, index=df.index)

    def _calc_shape_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        returns = self._temp_cache.get("returns")
        if returns is None:
            returns = df[self.price_column].astype(float).pct_change().fillna(0.0)
            self._temp_cache["returns"] = returns
        window = max(15, self.shape_window)
        rolling = returns.rolling(window, min_periods=window // 3)
        skew = rolling.skew().fillna(0.0)
        kurt = rolling.kurt().fillna(0.0)
        return pd.DataFrame({"skew": skew, "kurtosis": kurt})

    # ------------------------------------------------------------------
    # Regime labelling and normalization
    # ------------------------------------------------------------------

    def _annotate_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        regime, vol_bucket, realized_vol = _compute_regime_labels(
            df,
            price_column=self.price_column,
            vol_window=self.regime_vol_window,
            trend_window=self.regime_trend_window,
            vol_threshold=self.regime_vol_threshold,
            volatility_bins=self.volatility_bins,
        )
        df["regime"] = regime
        df["vol_bucket"] = vol_bucket
        df["volatility_realized"] = realized_vol
        return df

    def _apply_normalization(self, df: pd.DataFrame, *, split: str, refresh: bool) -> pd.DataFrame:
        columns = self._select_normalizable_columns(df)
        if not columns:
            return df
        if split == "train":
            accumulator = self._load_normalization_accumulator()
            if refresh or accumulator is None:
                accumulator = NormalizationAccumulator(strategy=self.normalization_strategy)
            accumulator.update(df, columns)
            self._normalization_accumulator = accumulator
            self._save_normalization_accumulator(accumulator)
            stats = accumulator.statistics()
        else:
            accumulator = self._load_normalization_accumulator()
            if accumulator is None:
                raise RuntimeError(
                    f"Normalization statistics missing at {self._normalization_path}; fit on train split first."
                )
            stats = accumulator.statistics()
        for col in columns:
            mean, std = stats.get(col, (0.0, 1.0))
            if std == 0.0:
                std = 1.0
            df[col] = (df[col].astype(float) - mean) / std
        return df

    # ------------------------------------------------------------------
    # Monitoring and persistence
    # ------------------------------------------------------------------

    def _collect_monitoring_metrics(
        self,
        df: pd.DataFrame,
        quality_metrics: Mapping[str, float],
        sync_metrics: Mapping[str, float],
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "rows": float(len(df)),
            "schema_version": float(SCHEMA_VERSION.replace(".", "")[:3]),
        }
        metrics.update({f"quality_{k}": float(v) for k, v in quality_metrics.items()})
        metrics.update({f"sync_{k}": float(v) for k, v in sync_metrics.items()})
        metrics["quality_gaps_rate"] = (
            metrics.get("quality_gap_count", 0.0) / metrics["rows"] if metrics["rows"] else 0.0
        )
        spread = df.get("spread")
        mid = df.get("mid")
        if spread is not None and mid is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                spread_bps = (spread / mid.replace(0, np.nan)) * 10_000
            metrics["p99_spread_bps"] = float(np.nanpercentile(spread_bps.fillna(0.0), 99))
        else:
            metrics["p99_spread_bps"] = 0.0
        depth_bid = self._collect_depth_levels(df, "bid").sum(axis=1)
        depth_ask = self._collect_depth_levels(df, "ask").sum(axis=1)
        depth_change = (depth_bid + depth_ask).diff().abs().fillna(0.0)
        metrics["p99_depth_change"] = float(np.nanpercentile(depth_change, 99)) if not depth_change.empty else 0.0

        returns = self._temp_cache.get("returns")
        mid_returns = self._temp_cache.get("mid_returns")
        if returns is None or mid_returns is None or returns.empty or mid_returns.empty:
            metrics["best_lag_sec"] = 0.0
            metrics["corr_at_best_lag"] = 0.0
        else:
            lags = range(-5, 6)
            best_corr = -1.0
            best_lag = 0
            median_interval = float(
                df[self.ts_column].diff().dt.total_seconds().median() or 1.0
            )
            for lag in lags:
                if lag < 0:
                    shifted = returns.shift(-lag)
                    corr = shifted.corr(mid_returns)
                elif lag > 0:
                    shifted = mid_returns.shift(lag)
                    corr = returns.corr(shifted)
                else:
                    corr = returns.corr(mid_returns)
                if corr is None:
                    continue
                if corr > best_corr:
                    best_corr = float(corr)
                    best_lag = lag
            metrics["best_lag_sec"] = float(best_lag * median_interval)
            metrics["corr_at_best_lag"] = float(best_corr if best_corr != -1.0 else 0.0)
        return metrics

    def _log_summary(self, df: pd.DataFrame, metrics: Mapping[str, float]) -> None:
        logger.info(
            "Feature pipeline summary: ticks={}, gaps_rate={:.6f}, best_lag={:.2f}s corr={:.3f}, p99_spread_bps={:.2f}",
            int(metrics.get("rows", 0.0)),
            metrics.get("quality_gaps_rate", 0.0),
            metrics.get("best_lag_sec", 0.0),
            metrics.get("corr_at_best_lag", 0.0),
            metrics.get("p99_spread_bps", 0.0),
        )

    def _export_metrics(self, df: pd.DataFrame, metrics: Mapping[str, float]) -> None:
        ts = self._ensure_datetime(df[self.ts_column])
        start_ts = ts.iloc[0].astimezone(timezone.utc)
        end_ts = ts.iloc[-1].astimezone(timezone.utc)
        file_name = f"metrics_{start_ts.strftime('%Y%m%dT%H%M%S')}_{end_ts.strftime('%Y%m%dT%H%M%S')}.json"
        payload = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "start_ts": start_ts.isoformat(),
            "end_ts": end_ts.isoformat(),
            "metrics": dict(metrics),
        }
        (self.monitoring_dir / file_name).write_text(json.dumps(payload, indent=2))

    def _cache_raw_data(self, frame: pd.DataFrame, ts_series: pd.Series) -> None:
        if self.raw_cache_dir is None:
            return
        df = frame.copy()
        df[self.ts_column] = ts_series
        partition_series = df[self.ts_column].dt.tz_convert(timezone.utc).dt.floor("D")
        for part, chunk in df.groupby(partition_series):
            partition_dir = self.raw_cache_dir / f"date={part.strftime('%Y-%m-%d')}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            file_path = partition_dir / "raw.parquet"
            chunk.to_parquet(
                file_path,
                index=False,
                engine="pyarrow",
                compression="zstd",
                use_dictionary=True,
            )

    def _materialize_features(
        self,
        df: pd.DataFrame,
        schema_signature: List[Tuple[str, str]],
        schema_hash: str,
    ) -> None:
        if self.feature_store_dir is None:
            return
        metadata_path = self.feature_store_dir / "metadata.json"
        metadata = self._load_metadata(metadata_path)
        metadata["schema_version"] = SCHEMA_VERSION
        metadata["pipeline_version"] = self.pipeline_version
        metadata["feature_schema_hash"] = schema_hash
        metadata["schema_signature"] = schema_signature
        metadata["transforms"] = [t.name for t in self.transforms]
        metadata["normalization"] = {
            "strategy": self.normalization_strategy,
            "path": str(self._normalization_path),
        }
        metadata.setdefault("partitions", {})

        ts = df[self.ts_column].dt.tz_convert(timezone.utc)
        partitions = ts.dt.to_period(self.materialization_freq)
        for period, chunk in df.groupby(partitions):
            part_dir = self.feature_store_dir / f"partition={period}"
            part_dir.mkdir(parents=True, exist_ok=True)
            file_path = part_dir / "features.parquet"
            chunk.to_parquet(
                file_path,
                index=False,
                engine="pyarrow",
                compression="zstd",
                use_dictionary=True,
            )
            coverage = self._regime_coverage(chunk, partition=str(period))
            metadata["partitions"][str(period)] = {
                "rows": int(len(chunk)),
                "start_ts": chunk[self.ts_column].iloc[0].isoformat(),
                "end_ts": chunk[self.ts_column].iloc[-1].isoformat(),
                "regime_counts": coverage["regime_counts"],
                "vol_bucket_counts": coverage["vol_bucket_counts"],
                "regime_vol_matrix": coverage["regime_vol_matrix"],
            }
        metadata["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_depth_levels(self, df: pd.DataFrame, side: str) -> pd.DataFrame:
        side = side.lower()
        qty_prefix = f"{side}_qty"
        columns = [col for col in df.columns if col.startswith(qty_prefix)]
        level_map: Dict[int, pd.Series] = {}
        for col in columns:
            suffix = col[len(qty_prefix) :]
            if suffix.startswith("_"):
                suffix = suffix[1:]
            level = 1
            try:
                level = int(suffix) if suffix else 1
            except ValueError:
                level = 1
            level_map[level] = df[col].astype(float)
        if not level_map:
            try:
                fallback = self._resolve_series(
                    df,
                    self.bid_size_columns if side == "bid" else self.ask_size_columns,
                )
            except KeyError:
                fallback = pd.Series(0.0, index=df.index)
            level_map[1] = fallback.astype(float)
        series_dict = {level: level_map[level] for level in sorted(level_map)}
        return pd.DataFrame(series_dict).reindex(df.index, fill_value=0.0)

    def _regime_coverage(self, df: pd.DataFrame, partition: str = "overall") -> Dict[str, object]:
        regime_counts = df["regime"].value_counts(dropna=False).to_dict() if "regime" in df.columns else {}
        vol_counts = df["vol_bucket"].value_counts(dropna=False).to_dict() if "vol_bucket" in df.columns else {}
        matrix: Dict[str, Dict[str, int]] = defaultdict(dict)
        if "regime" in df.columns and "vol_bucket" in df.columns:
            grouped = df.groupby(["regime", "vol_bucket"]).size()
            for (regime, bucket), count in grouped.items():
                matrix[str(regime)][str(bucket)] = int(count)
        return {
            "partition": partition,
            "regime_counts": {str(k): int(v) for k, v in regime_counts.items()},
            "vol_bucket_counts": {str(k): int(v) for k, v in vol_counts.items()},
            "regime_vol_matrix": {reg: dict(buckets) for reg, buckets in matrix.items()},
        }

    def _select_normalizable_columns(self, df: pd.DataFrame) -> List[str]:
        columns: List[str] = []
        for col in df.columns:
            if col == self.ts_column or col == "symbol":
                continue
            if any(col.startswith(prefix) for prefix in self.diagnostics_prefixes):
                continue
            if col in self.exclude_from_normalization:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                columns.append(col)
        return columns

    def _load_normalization_accumulator(self) -> NormalizationAccumulator | None:
        if self._normalization_accumulator is not None:
            return self._normalization_accumulator
        if not self._normalization_path.exists():
            return None
        payload = json.loads(self._normalization_path.read_text())
        self._normalization_accumulator = NormalizationAccumulator.from_json(payload)
        return self._normalization_accumulator

    def _save_normalization_accumulator(self, accumulator: NormalizationAccumulator) -> None:
        self._normalization_path.write_text(json.dumps(accumulator.to_json(), indent=2))

    def _compute_schema_signature(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        return [(str(col), str(dtype)) for col, dtype in df.dtypes.items()]

    def _ensure_schema_compatibility(
        self,
        signature: List[Tuple[str, str]],
        schema_hash: str,
    ) -> None:
        if self.feature_store_dir is None:
            return
        metadata_path = self.feature_store_dir / "metadata.json"
        if not metadata_path.exists():
            return
        payload = json.loads(metadata_path.read_text())
        existing_hash = payload.get("feature_schema_hash")
        if existing_hash and existing_hash != schema_hash:
            raise RuntimeError(
                f"Feature schema mismatch detected. expected={existing_hash} got={schema_hash}"
            )
        existing_signature = payload.get("schema_signature")
        if existing_signature and existing_signature != signature:
            raise RuntimeError("Feature schema signature changed; aborting to prevent data leak.")

    def _load_metadata(self, path: Path) -> Dict[str, object]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("Corrupted metadata detected at %s; recreating.", path)
            return {}

    def _prepare_market_record(self, record: Mapping[str, object]) -> Dict[str, object]:
        ts_raw = record.get(self.ts_column)
        timestamp = pd.Timestamp(ts_raw, tz=self.tz) if ts_raw is not None else pd.Timestamp(0, tz=self.tz)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(self.tz)
        timestamp = timestamp.tz_convert(timezone.utc)
        bid_px = self._extract_value(record, self.bid_price_columns)
        ask_px = self._extract_value(record, self.ask_price_columns)
        bids = self._extract_depth(record, side="bid", price=bid_px)
        asks = self._extract_depth(record, side="ask", price=ask_px)
        symbol = str(record.get("symbol") or record.get("sym") or "")
        last_trade = record.get("last_trade")
        return {
            "ts": int(timestamp.timestamp() * 1000),
            "symbol": symbol,
            "bid1": bid_px,
            "ask1": ask_px,
            "bids": bids,
            "asks": asks,
            "last_trade": last_trade,
        }

    def _extract_depth(
        self,
        record: Mapping[str, object],
        *,
        side: str,
        price: float,
    ) -> List[Tuple[float, float]]:
        key = "bids" if side == "bid" else "asks"
        raw = record.get(key)
        if raw is not None:
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raw = None
            if isinstance(raw, Iterable):
                result: List[Tuple[float, float]] = []
                for level in raw:
                    try:
                        lvl_price = float(level[0])
                        qty = float(level[1])
                        result.append((lvl_price, qty))
                    except (TypeError, ValueError, IndexError):
                        continue
                if result:
                    return result
        levels = self._collect_depth_levels(pd.DataFrame([record]), side=side)
        prices = self._collect_depth_prices(pd.DataFrame([record]), side=side)
        result = []
        for level in sorted(levels.columns):
            qty = float(levels[level].iloc[0])
            lvl_price = float(prices.get(level, price))
            result.append((lvl_price, qty))
        if not result:
            qty = float(self._extract_value(record, self.bid_size_columns if side == "bid" else self.ask_size_columns))
            if not np.isnan(price) and not np.isnan(qty):
                result.append((price, qty))
        return result

    def _collect_depth_prices(self, df: pd.DataFrame, side: str) -> Dict[int, float]:
        side = side.lower()
        px_prefix = f"{side}_px"
        result: Dict[int, float] = {}
        for col in df.columns:
            if not col.startswith(px_prefix):
                continue
            suffix = col[len(px_prefix) :]
            if suffix.startswith("_"):
                suffix = suffix[1:]
            level = 1
            try:
                level = int(suffix) if suffix else 1
            except ValueError:
                level = 1
            result[level] = float(df[col].astype(float).iloc[0])
        if not result and f"{side}_px" in df.columns:
            result[1] = float(df[f"{side}_px"].astype(float).iloc[0])
        return result

    def _mid_price_series(self, df: pd.DataFrame) -> pd.Series:
        bid_px = self._resolve_series(df, self.bid_price_columns)
        ask_px = self._resolve_series(df, self.ask_price_columns)
        return 0.5 * (bid_px + ask_px)

    def _resolve_series(self, df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
        for col in candidates:
            if col in df.columns:
                return df[col].astype(float)
        raise KeyError(f"None of the candidate columns found: {candidates}")

    def _extract_value(self, record: Mapping[str, object], candidates: Sequence[str]) -> float:
        for col in candidates:
            value = record.get(col)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return float("nan")

    def _ensure_datetime(self, series: pd.Series) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(series):
            if series.dt.tz is None:
                return series.dt.tz_localize(self.tz)
            return series.dt.tz_convert(self.tz)
        converted = pd.to_datetime(series, utc=True, errors="coerce")
        return converted.dt.tz_convert(self.tz)


def _compute_regime_labels(
    df: pd.DataFrame,
    *,
    price_column: str,
    vol_window: int,
    trend_window: int,
    vol_threshold: float,
    volatility_bins: Sequence[float],
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    price = df[price_column].astype(float)
    returns = price.pct_change().fillna(0.0)
    realized_vol = returns.rolling(vol_window, min_periods=max(5, vol_window // 4)).std().fillna(0.0)
    trend = price.rolling(trend_window, min_periods=max(5, trend_window // 4)).apply(
        lambda x: x[-1] - x[0],
        raw=True,
    )
    trend = trend.fillna(0.0)
    regimes = pd.Series("range", index=df.index, dtype="object")
    vol_cut = vol_threshold or 0.002
    bullish = (trend > 0) & (realized_vol >= vol_cut)
    bearish = (trend < 0) & (realized_vol >= vol_cut)
    regimes.loc[bullish] = "bull"
    regimes.loc[bearish] = "bear"
    bins = list(volatility_bins) if volatility_bins else [vol_cut]
    labels = ["low", "medium", "high"]
    if len(bins) == 1:
        quantiles = [bins[0]]
    elif len(bins) >= 2:
        quantiles = [bins[0], bins[1]]
    else:
        quantiles = [vol_cut]
    vol_bucket = pd.cut(
        realized_vol,
        bins=[-np.inf] + quantiles + [np.inf],
        labels=labels[: len(quantiles) + 1],
    ).astype("object")
    vol_bucket = vol_bucket.fillna("low")
    return regimes, vol_bucket, realized_vol


def split_by_regime(
    df: pd.DataFrame,
    *,
    price_column: str = "price",
    vol_window: int = 300,
    trend_window: int = 900,
    vol_threshold: float = 0.002,
    volatility_bins: Sequence[float] = DEFAULT_VOL_BUCKETS,
) -> Dict[str, pd.DataFrame]:
    if df.empty:
        return {"bull": df.copy(), "bear": df.copy(), "range": df.copy()}
    regimes, vol_bucket, realized_vol = _compute_regime_labels(
        df,
        price_column=price_column,
        vol_window=vol_window,
        trend_window=trend_window,
        vol_threshold=vol_threshold,
        volatility_bins=volatility_bins,
    )
    enriched = df.copy()
    enriched["regime"] = regimes
    enriched["vol_bucket"] = vol_bucket
    enriched["volatility_realized"] = realized_vol
    result: Dict[str, pd.DataFrame] = {}
    for regime in ("bull", "bear", "range"):
        mask = enriched["regime"] == regime
        result[regime] = enriched.loc[mask].copy()
    return result
