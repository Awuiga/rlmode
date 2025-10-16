"""End-to-end pipeline: download historical data, build scalper dataset splits."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from app.rl.data import (
    DatasetSplitConfig,
    HistoricalDataCollector,
    HistoricalDataConfig,
    MarketRegimeConfig,
    RLScalperDataset,
)


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


async def collect_and_build(args: argparse.Namespace) -> None:
    start = parse_dt(args.start)
    end = parse_dt(args.end)
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)

    hist_cfg = HistoricalDataConfig(
        symbol=args.symbol,
        start=start,
        end=end,
        output_dir=raw_dir,
        levels=args.depth,
        onchain_metrics=tuple(args.onchain_metrics),
        funding_symbol=args.funding_symbol or f"{args.symbol}USDT",
    )
    collector = HistoricalDataCollector(hist_cfg)
    await collector.run()

    parquet_paths = sorted(raw_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no parquet files generated in {raw_dir}")

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    split_cfg = DatasetSplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
        micro_window=args.micro_window,
        macro_window=args.macro_window,
    )
    regime_cfg = MarketRegimeConfig(
        return_window=args.regime_return_window,
        volatility_window=args.regime_vol_window,
        bull_return_threshold=args.bull_threshold,
        bear_return_threshold=-abs(args.bear_threshold),
        max_range_volatility=args.range_volatility,
    )
    dataset = RLScalperDataset(
        parquet_paths=parquet_paths,
        output_root=processed_dir,
        split_cfg=split_cfg,
        regime_cfg=regime_cfg,
    )
    artifacts = dataset.build()
    print("Generated datasets:")
    for split, path in artifacts.items():
        print(f"  {split}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare RL scalper dataset")
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. BTCUSDT")
    parser.add_argument("--start", required=True, help="ISO8601 start datetime (UTC).")
    parser.add_argument("--end", required=True, help="ISO8601 end datetime (UTC).")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--onchain-metrics", nargs="*", default=[])
    parser.add_argument("--funding-symbol", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--micro-window", type=int, default=60)
    parser.add_argument("--macro-window", type=int, default=60)
    parser.add_argument("--regime-return-window", type=int, default=60)
    parser.add_argument("--regime-vol-window", type=int, default=120)
    parser.add_argument("--bull-threshold", type=float, default=0.01)
    parser.add_argument("--bear-threshold", type=float, default=0.01)
    parser.add_argument("--range-volatility", type=float, default=0.005)
    args = parser.parse_args()

    asyncio.run(collect_and_build(args))


if __name__ == "__main__":
    main()
