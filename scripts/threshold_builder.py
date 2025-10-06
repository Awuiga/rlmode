#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np

from train import compute_threshold_sweep, select_thresholds


def build_thresholds(df: pd.DataFrame, group_col: str, precision_target: float) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for value, chunk in df.groupby(group_col):
        if 'meta_label' not in chunk.columns:
            continue
        y_true = chunk['meta_label'].astype(int).to_numpy()
        scores = chunk['p_calibrated'].astype(float).to_numpy() if 'p_calibrated' in chunk.columns else chunk['p_success'].astype(float).to_numpy()
        sweep = compute_threshold_sweep(y_true, scores)
        entry, hold = select_thresholds(
            sweep,
            precision_target=precision_target,
            min_support=max(10, int(len(chunk) * 0.05)),
            min_avg_markout=0.0,
            hold_precision_ratio=0.9,
        )
        results[str(value)] = {
            'tau_entry': float(entry['threshold']),
            'tau_hold': float(min(entry['threshold'], hold['threshold'])),
            'precision': float(entry['precision']),
            'support': float(entry['support']),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Build regime-specific thresholds from labelled dataset')
    parser.add_argument('--data', required=True, help='Dataset with meta_label and p_success columns')
    parser.add_argument('--group', default='spread_regime', help='Column to group by (e.g., spread_regime)')
    parser.add_argument('--precision-target', type=float, default=0.6)
    parser.add_argument('--output', default='reports/thresholds_by_regime.json')
    args = parser.parse_args()

    df = pd.read_parquet(args.data) if args.data.endswith(('parquet', 'pq')) else pd.read_csv(args.data)
    if args.group not in df.columns:
        raise SystemExit(f'group column {args.group} missing')
    results = build_thresholds(df, args.group, args.precision_target)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f'saved thresholds to {args.output}')


if __name__ == '__main__':
    main()
