from __future__ import annotations

import os
import re
from dataclasses import field
from typing import List, Optional, Dict, Any, Tuple

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
    # Back-compat fields (deprecated): prefer market.use and market.symbols
    exchange: Optional[str] = None
    symbols: List[str] = Field(default_factory=list)
    redis: RedisConfig
    collector: CollectorConfig = Field(default_factory=CollectorConfig)
    signal_engine: SignalEngineConfig = Field(default_factory=SignalEngineConfig)
    ai_scorer: AIScorerConfig = Field(default_factory=AIScorerConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    # New unified controls
    exchange_mode: str = Field(default="live_paper")  # live_paper | live_real
    
    class MarketConfig(BaseModel):
        use: str = Field(default="binance_futs")  # binance_futs | bybit_v5 | fake
        symbols: List[str] = Field(default_factory=list)
        binance_ws_public: str = Field(default="wss://fstream.binance.com/stream")
        bybit_ws_public: str = Field(default="wss://stream.bybit.com/v5/public")

    class SimulatorCurve(BaseModel):
        x: List[float] = Field(default_factory=lambda: [10000, 50000, 200000, 1000000])
        y_bps: List[float] = Field(default_factory=lambda: [1.5, 3.0, 6.0, 20.0])

    class ExecutionSimulatorConfig(BaseModel):
        seed: int = 42
        base_latency_ms: int = 15
        latency_jitter_ms: List[int] = Field(default_factory=lambda: [2, 8])
        maker_fill_model: str = Field(default="queue")  # queue | poisson
        taker_slip_bps: List[float] = Field(default_factory=lambda: [1.0, 4.0])
        maker_fill_boost_if_qi: float = 0.15
        min_partial_fill_qty: float = 0.25
        cancel_timeout_ms: int = 800
        shock_prob: float = 0.0005
        shock_return: List[float] = Field(default_factory=lambda: [-0.03, -0.10])
        liquidity_degrade_curve: 'AppConfig.SimulatorCurve' = Field(default_factory=lambda: AppConfig.SimulatorCurve())
        fee_bps_maker: float = 0.0
        allow_taker_on_timeout: bool = False

    class ExecutionConfig(BaseModel):
        mode: str = Field(default="paper")  # paper | real
        simulator: 'AppConfig.ExecutionSimulatorConfig' = Field(default_factory=lambda: AppConfig.ExecutionSimulatorConfig())

    class SignalsConfig(BaseModel):
        tp_pct: float = 0.006
        sl_pct: float = 0.003
        score_threshold: float = 0.65
        sigma_window_ms: int = 1500
        sigma_min: float = 0.0004
        sigma_max: float = 0.002
        ofi_thr: float = 0.10
        qi_thr: float = 0.05
        ladder: List[float] = Field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
        ladder_step_ticks: int = 1

    market: MarketConfig = Field(default_factory=MarketConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    signals: SignalsConfig = Field(default_factory=SignalsConfig)


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
        cfg = AppConfig.model_validate(expanded)
        # Back-compat: mirror market/use and market/symbols into legacy fields if missing
        if not cfg.exchange:
            cfg.exchange = cfg.market.use
        if not cfg.symbols:
            cfg.symbols = list(cfg.market.symbols)
        # Also align signal_engine high level knobs from signals if present
        if cfg.signals:
            cfg.signal_engine.tp_pct = cfg.signals.tp_pct
            cfg.signal_engine.sl_pct = cfg.signals.sl_pct
            cfg.signal_engine.features.sigma_window_ms = cfg.signals.sigma_window_ms
            cfg.signal_engine.rules.sigma_min = cfg.signals.sigma_min
            cfg.signal_engine.rules.sigma_max = cfg.signals.sigma_max
            cfg.signal_engine.rules.qi_min = cfg.signals.qi_thr
            # scoring threshold
            cfg.signal_engine.scoring.theta_emit = cfg.signals.score_threshold
            # ladder setup
            cfg.executor.ladder.fractions = cfg.signals.ladder
            cfg.executor.ladder.step_ticks = cfg.signals.ladder_step_ticks
        return cfg
    except ValidationError as e:
        raise SystemExit(f"Invalid configuration {cfg_path}: {e}")
