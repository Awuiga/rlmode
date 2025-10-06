#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import yaml


def load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate symbol metadata against reference values')
    parser.add_argument('--symbols', default='config/symbols.yml', help='Path to symbols configuration')
    parser.add_argument('--reference', default='config/exchange_reference.yml', help='Reference symbol definitions')
    parser.add_argument('--output', default='reports/symbol_validation.json')
    args = parser.parse_args()

    symbols = load_yaml(Path(args.symbols))
    reference = load_yaml(Path(args.reference))
    issues = []
    for symbol, params in symbols.items():
        ref = reference.get(symbol)
        if not ref:
            issues.append({'symbol': symbol, 'issue': 'missing_in_reference'})
            continue
        for key in ('tick_size', 'lot_step', 'maker_fee', 'taker_fee'):
            if key in params and key in ref and float(params[key]) != float(ref[key]):
                issues.append({'symbol': symbol, 'field': key, 'expected': ref[key], 'actual': params[key]})
    report = {'issues': issues}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    if issues:
        print(f'found {len(issues)} symbol discrepancies -> {args.output}')
        raise SystemExit(f'symbol validation failed with {len(issues)} discrepancies')
    print('all symbols match reference')


if __name__ == '__main__':
    main()
