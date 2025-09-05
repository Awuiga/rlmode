from __future__ import annotations

import os
import re
from dataclasses import field
from typing import List, Optional, Dict, Any

import yaml
from pydantic import BaseModel, Field, ValidationError


class RedisConfig(BaseModel):
    url: str = Field(..., description="Redis connection URL")


class CollectorWSConfig(BaseModel):
    binance: Dict[str, Any] = Field(default_factory=dict)
    bybit: Dict[str, Any] = Field(default_factory=dict)


class CollectorConfig(BaseModel):
    depth_levels: int = 10
    ws: CollectorWSConfig = Field(default_factory=CollectorWSConfig)
    reconnect_backoff_ms: List[int] = Field(default_factory=lambda: [500, 1000, 2000, 5000])
    rate_limit_per_sec: int = 20


class FeatureConfig(BaseModel):
    depth_top_k: int = 5
    sigma_window_ms: int = 1500
    ofi_window_ms: int = 1000
    ema_window_ms: int = 2000


class RulesConfig(BaseModel):
    min_liquidity: float = 50.0
    max_spread_ticks: int = 3
    sigma_min: float = 0.0004
    sigma_max: float = 0.004
    ofi_buy_min: float = 5.0
    ofi_sell_min: float = 5.0
    qi_min: float = 0.1
    use_trend: bool = False


class ScoringWeights(BaseModel):
    spread: float = -0.2
    sigma: float = 0.3
    ofi: float = 0.3
    qi: float = 0.2
    depth: float = 0.2


class ScoringConfig(BaseModel):
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    theta_emit: float = 0.6


class SignalEngineConfig(BaseModel):
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    tp_pct: float = 0.006
    sl_pct: float = 0.003


class AIScorerWindow(BaseModel):
    frequency_hz: int = 50
    seconds: int = 2
    depth_levels: int = 10


class AIScorerConfig(BaseModel):
    model_path: str = "/app/models/seq_model.onnx"
    threshold: float = 0.55
    batch_size: int = 32
    window: AIScorerWindow = Field(default_factory=AIScorerWindow)


class LadderConfig(BaseModel):
    fractions: List[float] = Field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    step_ticks: int = 2
    use_percent: bool = False


class ExecutorConfig(BaseModel):
    ladder: LadderConfig = Field(default_factory=LadderConfig)
    tif: str = "GTC"
    post_only: bool = True
    cancel_timeout_ms: int = 2500
    retry_backoff_ms: List[int] = Field(default_factory=lambda: [200, 400, 800, 1600])
    adaptive_stop_k: float = 2.0
    min_stop: float = 0.002


class MonitorConfig(BaseModel):
    http_host: str = "0.0.0.0"
    http_port: int = 8000


class RiskConfig(BaseModel):
    enable: bool = True


class AppConfig(BaseModel):
    exchange: str
    symbols: List[str]
    redis: RedisConfig
    collector: CollectorConfig = Field(default_factory=CollectorConfig)
    signal_engine: SignalEngineConfig = Field(default_factory=SignalEngineConfig)
    ai_scorer: AIScorerConfig = Field(default_factory=AIScorerConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)


ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.+?))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match[str]):
            name = m.group(1)
            default = m.group(2)
            return os.environ.get(name, default or "")

        return ENV_VAR_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_app_config(path: Optional[str] = None) -> AppConfig:
    cfg_path = path or os.environ.get("APP_CONFIG", "config/app.yml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    expanded = _expand_env(raw)
    try:
        return AppConfig.model_validate(expanded)
    except ValidationError as e:
        raise SystemExit(f"Invalid configuration {cfg_path}: {e}")

