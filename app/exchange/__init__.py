from __future__ import annotations

from typing import Dict

from ..common.config import AppConfig
from ..common.redis_stream import RedisStream
from .paper import PaperExchange
from .real import RealExchange


__all__ = ["create_exchange", "PaperExchange", "RealExchange"]


def create_exchange(cfg: AppConfig, rs: RedisStream, symbols_meta: Dict[str, Dict[str, float]]):
    """Instantiate the proper exchange adapter based on runtime configuration."""
    mode = (cfg.exchange.mode or "paper").lower()
    if mode == "real":
        return RealExchange(
            venue=cfg.exchange.venue,
            maker_only=cfg.exchange.maker_only,
            reduce_only=cfg.exchange.reduce_only,
            post_only=cfg.exchange.post_only,
        )
    # Default to paper simulator
    return PaperExchange(cfg=cfg.execution.simulator, rs=rs, symbols_meta=symbols_meta)
