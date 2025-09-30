#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def scan_directory(root: Path, threshold_ms: int) -> List[Dict[str, object]]:
    issues: List[Dict[str, object]] = []
    for path in root.rglob('*.parquet'):
        df = pd.read_parquet(path, columns=['ts_ms', 'symbol'])
        df = df.sort_values('ts_ms')
        ts = df['ts_ms'].to_numpy()
        gaps = ts[1:] - ts[:-1]
        bad = gaps > threshold_ms
        for idx in bad.nonzero()[0]:
            issues.append({
                'file': str(path),
                'symbol': df['symbol'].iloc[idx + 1],
                'gap_ms': int(gaps[idx]),
                'ts_start': int(ts[idx]),
                'ts_end': int(ts[idx + 1]),
            })
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description='Scan parquet directory for temporal gaps')
    parser.add_argument('--root', default='data/parquet', help='Root directory containing parquet partitions')
    parser.add_argument('--threshold-ms', type=int, default=60000, help='Gap threshold in milliseconds')
    parser.add_argument('--output', default='reports/parquet_gaps.json')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f'{root} does not exist')
    issues = scan_directory(root, args.threshold_ms)
    Path(args.output).write_text(json.dumps({'issues': issues}, indent=2))
    print(f'scanned {root}, found {len(issues)} gaps')


if __name__ == '__main__':
    main()
