#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def compute_histogram(series: pd.Series, bins: int) -> Dict[str, float]:
    counts, edges = np.histogram(series.to_numpy(dtype=float), bins=bins)
    total = counts.sum() or 1
    centres = (edges[:-1] + edges[1:]) / 2.0
    return {f"bin_{idx}": float(count / total) for idx, count in enumerate(counts)}


def calibrate(path: Path, output: Path, feature: str, depth_col: str) -> None:
    if path.suffix in {'.parquet', '.pq'}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    required = {feature, depth_col, 'filled'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns: {missing}")
    df['filled'] = df['filled'].astype(int)
    grouped = df.groupby(pd.qcut(df[feature], q=10, duplicates='drop'))
    rows: List[Dict[str, float]] = []
    for interval, chunk in grouped:
        prob = chunk['filled'].mean()
        depth_mean = chunk[depth_col].mean()
        rows.append({
            'feature_bin': str(interval),
            'fill_probability': float(prob),
            'avg_depth': float(depth_mean),
            'count': float(len(chunk)),
        })
    histogram = compute_histogram(df[feature], bins=20)
    payload = {
        'source': str(path),
        'feature': feature,
        'depth_column': depth_col,
        'histogram': histogram,
        'calibration_table': rows,
    }
    output.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate empirical fill probability curves")
    parser.add_argument('--fills', required=True, help='Path to fills dataset (CSV or Parquet)')
    parser.add_argument('--feature', default='queue_position_proxy', help='Column representing queue proxy / QI')
    parser.add_argument('--depth-col', default='depth', help='Column representing top-of-book depth')
    parser.add_argument('--output', default='reports/fill_calibration.json', help='Where to write calibration table')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibrate(Path(args.fills), output_path, args.feature, args.depth_col)
    print(f"saved calibration to {output_path}")


if __name__ == '__main__':
    main()
