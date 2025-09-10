from __future__ import annotations

from app.common.config import load_app_config


def test_config_load_defaults():
    cfg = load_app_config()
    assert cfg.redis.url
    assert cfg.market.use in ("binance_futs", "bybit_v5", "fake")
    assert isinstance(cfg.symbols, list)
