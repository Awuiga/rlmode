#!/usr/bin/env python3
from __future__ import annotations

import argparse

from app.common.config import AppConfig, load_app_config


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate application configuration using Pydantic schema')
    parser.add_argument('--config', default='config/app.yml')
    args = parser.parse_args()

    try:
        cfg = load_app_config(args.config)
    except SystemExit as exc:
        raise SystemExit(f'configuration invalid: {exc}')
    summary = {
        'symbols': cfg.symbols,
        'redis': cfg.redis.url,
        'executor_mode': cfg.execution.mode,
    }
    print('configuration valid:', summary)


if __name__ == '__main__':
    main()
