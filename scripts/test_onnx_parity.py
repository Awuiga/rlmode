#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate ONNX parity log reports zero failures')
    parser.add_argument('--log', default='ci_artifacts/onnx_parity.log', help='Path to ONNX parity log')
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f'{log_path} not found')
    contents = log_path.read_text().strip()
    if 'onnx_parity_ok' in contents:
        print(f'onnx parity ok -> {log_path}')
        return
    raise SystemExit(f'onnx parity failure detected -> {contents}')


if __name__ == '__main__':
    main()
