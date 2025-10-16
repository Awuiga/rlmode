"\"\"\"Historical data collection pipeline for RL scalper.\"\"\""

from __future__ import annotations

import asyncio
import io
import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import httpx

from ...features import FeaturePipeline, split_by_regime
from ...common.config import load_app_config
from ..data.fetchers import (
    BinanceFundingFetcher,
    BinanceSpotFetcher,
    NoopOnChainProvider,
    OnChainProvider,
    gather_concurrently,
)

BINANCE_VISION_BASE = "https://data.binance.vision/data/spot/daily"


@dataclass(slots=True)
class HistoricalDataConfig:
    symbol: str
    start: datetime
    end: datetime
    output_dir: Path
    levels: int = 20
    micro_window_seconds: int = 60
    macro_bar_seconds: int = 60
    funding_symbol: Optional[str] = None
    onchain_metrics: Sequence[str] = field(default_factory=tuple)
    exchange: str = "binance"
    feature_depth_top_k: int = 10
    chunk_days: int = 7
    tz: timezone = timezone.utc
    split: str = "train"


def daterange(start: datetime, end: datetime, step: timedelta) -> Iterable[datetime]:
    current = start
    while current <= end:
        yield current
        current += step


class HistoricalDataCollector:
    """Coordinator that harvests historical ticks, LOB snapshots and auxiliary factors."""

    def __init__(
        self,
        cfg: HistoricalDataConfig,
        *,
        spot_fetcher: Optional[BinanceSpotFetcher] = None,
        funding_fetcher: Optional[BinanceFundingFetcher] = None,
        onchain_provider: Optional[OnChainProvider] = None,
        feature_config_path: Path = Path("config/app.yml"),
    ) -> None:
        self.cfg = cfg
        self.output_dir = cfg.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_cfg = make_feature_state_config(feature_config_path)
        self.pipeline = FeaturePipeline(
            feature_config=self.feature_cfg,
            depth_top_k=self.cfg.feature_depth_top_k,
            ts_column="ts",
            price_column="price",
            raw_cache_dir=self.output_dir / "raw",
            feature_store_dir=self.output_dir / "features",
            progress=True,
        )
        self._normalization_initialized = False
        self.spot_fetcher = spot_fetcher or BinanceSpotFetcher(symbol=cfg.symbol)
        symbol_funding = cfg.funding_symbol or f"{cfg.symbol}USDT"
        self.funding_fetcher = funding_fetcher or BinanceFundingFetcher(symbol=symbol_funding)
        self.onchain_provider = onchain_provider or NoopOnChainProvider()

    async def run(self) -> None:
        tasks = []
        chunk_step = timedelta(days=self.cfg.chunk_days)
        for chunk_start in daterange(self.cfg.start, self.cfg.end, chunk_step):
            chunk_end = min(chunk_start + chunk_step - timedelta(seconds=1), self.cfg.end)
            tasks.append(asyncio.create_task(self._process_chunk(chunk_start, chunk_end)))
        await gather_concurrently(tasks)

    async def _process_chunk(self, chunk_start: datetime, chunk_end: datetime) -> None:
        trades_task = asyncio.create_task(self._collect_trades(chunk_start, chunk_end))
        lob_task = asyncio.create_task(self._collect_lob(chunk_start, chunk_end))
        funding_task = asyncio.create_task(self._collect_funding(chunk_start, chunk_end))
        onchain_task = asyncio.create_task(
            self._collect_onchain(chunk_start, chunk_end, self.cfg.onchain_metrics)
        )

        trades, lob_frames, funding, onchain = await asyncio.gather(
            trades_task, lob_task, funding_task, onchain_task
        )

        dataset = self._build_dataset(trades, lob_frames, funding, onchain, chunk_start, chunk_end)
        out_file = self._chunk_path(chunk_start, chunk_end)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(out_file, index=False)

    async def _collect_trades(self, start: datetime, end: datetime) -> pd.DataFrame:
        records: List[Dict[str, object]] = []
        async for trade in self.spot_fetcher.iter_trades(start, end):
            records.append(
                {
                    "ts": trade.timestamp,
                    "price": trade.price,
                    "qty": trade.quantity,
                    "is_buyer_maker": trade.is_buyer_maker,
                }
            )
        if not records:
            return pd.DataFrame(columns=["ts", "price", "qty", "is_buyer_maker"])
        trades = pd.DataFrame.from_records(records).sort_values("ts")
        trades["side"] = np.where(trades["is_buyer_maker"], "sell", "buy")
        return trades.reset_index(drop=True)

    async def _collect_lob(self, start: datetime, end: datetime) -> List[pd.DataFrame]:
        frames: List[pd.DataFrame] = []
        for day_start in daterange(start, end, timedelta(days=1)):
            day_end = min(day_start.replace(hour=23, minute=59, second=59, microsecond=0), end)
            frame = await download_binance_lob(
                symbol=self.cfg.symbol,
                day=day_start,
                levels=self.cfg.levels,
            )
            if frame.empty:
                continue
            mask = (frame["ts"] >= day_start) & (frame["ts"] <= day_end)
            frames.append(frame.loc[mask])
        return frames

    async def _collect_funding(self, start: datetime, end: datetime) -> pd.DataFrame:
        records: List[Dict[str, object]] = []
        async for rate in self.funding_fetcher.iter_funding_rates(start, end):
            records.append(
                {
                    "ts": rate.timestamp,
                    "funding_rate": rate.rate,
                }
            )
        frame = pd.DataFrame.from_records(records).sort_values("ts")
        return frame.reset_index(drop=True)

    async def _collect_onchain(
        self, start: datetime, end: datetime, metrics: Sequence[str]
    ) -> pd.DataFrame:
        if not metrics:
            return pd.DataFrame(columns=["ts"])
        records: List[Dict[str, object]] = []
        async for point in self.onchain_provider.iter_metrics(metrics, start, end):
            records.append(
                {
                    "ts": point.timestamp,
                    point.name: point.value,
                }
            )
        if not records:
            return pd.DataFrame(columns=["ts"] + list(metrics))
        frame = pd.DataFrame.from_records(records)
        frame = frame.groupby("ts").mean(numeric_only=True).sort_values("ts")
        return frame.reset_index()

    def _build_dataset(
        self,
        trades: pd.DataFrame,
        lob_frames: List[pd.DataFrame],
        funding: pd.DataFrame,
        onchain: pd.DataFrame,
        chunk_start: datetime,
        chunk_end: datetime,
    ) -> pd.DataFrame:
        if not lob_frames:
            raise ValueError("LOB frames are empty; ensure depth archives are available.")
        lob = pd.concat(lob_frames, ignore_index=True)
        lob = lob.sort_values("ts").reset_index(drop=True)
        trade_resampled = trades.set_index("ts").resample("1S").agg(
            {
                "price": "last",
                "qty": "sum",
                "side": lambda x: (x == "buy").sum() - (x == "sell").sum(),
            }
        )
        trade_resampled.rename(columns={"side": "order_imbalance"}, inplace=True)
        trade_resampled["order_imbalance"].fillna(0.0, inplace=True)
        trade_resampled["price"].ffill(inplace=True)
        trade_resampled["qty"].fillna(0.0, inplace=True)

        merged = lob.merge(
            trade_resampled.reset_index(), on="ts", how="left", validate="one_to_one"
        )
        merged["price"].ffill(inplace=True)
        merged["qty"].fillna(0.0, inplace=True)
        merged["order_imbalance"].fillna(0.0, inplace=True)

        if not funding.empty:
            merged = merged.merge(
                funding, on="ts", how="left"
            )
            merged["funding_rate"].ffill(inplace=True)
            merged["funding_rate"].fillna(0.0, inplace=True)
        else:
            merged["funding_rate"] = 0.0

        if not onchain.empty:
            merged = merged.merge(onchain, on="ts", how="left")
            merged.fillna(method="ffill", inplace=True)

        features = self._compute_features(merged, chunk_start, chunk_end)
        return features

    def _compute_features(
        self,
        merged: pd.DataFrame,
        chunk_start: datetime,
        chunk_end: datetime,
    ) -> pd.DataFrame:
        merged = merged.copy()
        merged["symbol"] = self.cfg.symbol
        refresh_stats = False
        if self.cfg.split == "train" and not getattr(self, "_normalization_initialized", False):
            refresh_stats = True
        enriched = self.pipeline.run(
            merged,
            symbol=self.cfg.symbol,
            cache_raw=True,
            materialize=True,
            split=self.cfg.split,
            refresh_stats=refresh_stats,
        )
        if self.cfg.split == "train":
            self._normalization_initialized = True
        micro_window = self.cfg.micro_window_seconds
        if "qty" in enriched.columns:
            enriched["cumulative_volume"] = (
                enriched["qty"].rolling(window=micro_window, min_periods=1).sum().fillna(0.0)
            )
        regimes = split_by_regime(
            enriched,
            price_column="price",
            vol_window=self.pipeline.regime_vol_window,
            trend_window=self.pipeline.regime_trend_window,
            vol_threshold=self.pipeline.regime_vol_threshold,
            volatility_bins=self.pipeline.volatility_bins,
        )
        self._materialize_regimes(regimes, chunk_start, chunk_end)
        return enriched

    def _materialize_regimes(
        self,
        regimes: Dict[str, pd.DataFrame],
        chunk_start: datetime,
        chunk_end: datetime,
    ) -> None:
        regime_dir = self.output_dir / "regimes"
        regime_dir.mkdir(parents=True, exist_ok=True)
        for regime, frame in regimes.items():
            if frame.empty:
                continue
            file_name = (
                f"{self.cfg.symbol}_{regime}_{chunk_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}.parquet"
            )
            output_path = regime_dir / file_name
            frame.to_parquet(
                output_path,
                index=False,
                engine="pyarrow",
                compression="zstd",
                use_dictionary=True,
            )
            metrics_dir = regime_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            vol_counts = frame.get("vol_bucket", pd.Series()).value_counts(dropna=False).to_dict()
            metrics_payload = {
                "regime": regime,
                "rows": int(len(frame)),
                "start_ts": frame["ts"].iloc[0].isoformat(),
                "end_ts": frame["ts"].iloc[-1].isoformat(),
                "vol_bucket_counts": {str(k): int(v) for k, v in vol_counts.items()},
            }
            metrics_name = file_name.replace(".parquet", "_metrics.json")
            (metrics_dir / metrics_name).write_text(json.dumps(metrics_payload, indent=2))

    def _chunk_path(self, chunk_start: datetime, chunk_end: datetime) -> Path:
        start_str = chunk_start.strftime("%Y%m%d")
        end_str = chunk_end.strftime("%Y%m%d")
        return self.output_dir / f"{self.cfg.symbol}_{start_str}_{end_str}.parquet"


def make_feature_state_config(config_path: Path):
    cfg = load_app_config(str(config_path))
    return cfg.features


async def download_binance_lob(*, symbol: str, day: datetime, levels: int) -> pd.DataFrame:
    day_local = day.strftime("%Y-%m-%d")
    filename = f"{symbol.upper()}-depth-{levels}-{day_local}.zip"
    url = f"{BINANCE_VISION_BASE}/depth/{symbol.upper()}/{filename}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.content

    def _parse_zip() -> pd.DataFrame:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            inner = next((name for name in zf.namelist() if name.endswith(".csv")), None)
            if inner is None:
                raise ValueError(f"zip archive {filename} missing csv")
            with zf.open(inner) as csv_file:
                columns = ["ts"]
                for level in range(1, levels + 1):
                    columns.extend(
                        [
                            f"bid_px_{level}",
                            f"bid_qty_{level}",
                            f"ask_px_{level}",
                            f"ask_qty_{level}",
                        ]
                    )
                df = pd.read_csv(csv_file, names=columns, header=None)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        if "bid_px" not in df.columns and "bid_px_1" in df.columns:
            df["bid_px"] = df["bid_px_1"]
        if "ask_px" not in df.columns and "ask_px_1" in df.columns:
            df["ask_px"] = df["ask_px_1"]
        if "bid_qty" not in df.columns and "bid_qty_1" in df.columns:
            df["bid_qty"] = df["bid_qty_1"]
        if "ask_qty" not in df.columns and "ask_qty_1" in df.columns:
            df["ask_qty"] = df["ask_qty_1"]
        df["spread"] = df["ask_px"] - df["bid_px"]
        df["mid"] = 0.5 * (df["ask_px"] + df["bid_px"])
        bid_qty_cols = [f"bid_qty_{level}" for level in range(1, levels + 1) if f"bid_qty_{level}" in df.columns]
        ask_qty_cols = [f"ask_qty_{level}" for level in range(1, levels + 1) if f"ask_qty_{level}" in df.columns]
        df["depth_bids"] = df[bid_qty_cols].sum(axis=1) if bid_qty_cols else df["bid_qty"]
        df["depth_asks"] = df[ask_qty_cols].sum(axis=1) if ask_qty_cols else df["ask_qty"]
        df["depth_imbalance"] = np.divide(
            df["depth_bids"] - df["depth_asks"],
            df["depth_bids"] + df["depth_asks"],
            out=np.zeros_like(df["depth_bids"], dtype=float),
            where=(df["depth_bids"] + df["depth_asks"]) != 0,
        )
        return df

    df = await asyncio.to_thread(_parse_zip)
    return df
