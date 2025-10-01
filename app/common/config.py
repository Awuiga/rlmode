from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from pydantic import BaseModel, Field, ValidationError, ConfigDict


class RedisRetryConfig(BaseModel):
    idle_ms: int = 5_000
    count: int = 50
    backoff_ms: List[int] = Field(default_factory=lambda: [200, 400, 800, 1_600])


class RedisConfig(BaseModel):
    url: str = Field(..., description="Redis connection URL")
    streams_maxlen: Optional[int] = None
    pending_retry: RedisRetryConfig = Field(default_factory=RedisRetryConfig)


class CollectorWSConfig(BaseModel):
    binance: Dict[str, Any] = Field(default_factory=dict)
    bybit: Dict[str, Any] = Field(default_factory=dict)


class CollectorConfig(BaseModel):
    depth_levels: int = 10
    ws: CollectorWSConfig = Field(default_factory=CollectorWSConfig)
    reconnect_backoff_ms: List[int] = Field(default_factory=lambda: [500, 1000, 2000, 5000])
    rate_limit_per_sec: int = 20



class ExchangeSettings(BaseModel):
    mode: str = "paper"
    venue: str = "bybit_v5"
    maker_only: bool = True
    reduce_only: bool = True
    post_only: bool = True


class RolloutConfig(BaseModel):
    mode: str = "disabled"
    canary_fraction: float = 0.10
    rng_seed: int = 1337


class RolloutGuardConfig(BaseModel):
    winrate_delta_pp: float = 5.0
    min_sample: int = 50
    psi_block: float = 0.30
    ks_block: float = 0.20
    cooldown_minutes: int = 15


class FeatureConfig(BaseModel):
    depth_top_k: int = 5
    qi_levels: int = 5
    sigma_window_ms: int = 1500
    ofi_window_ms: int = 1000
    ema_window_ms: int = 2000
    microprice_velocity_window_ms: int = 200
    cancel_window_ms: int = 2000
    taker_window_ms: int = 1000
    regime_window_ms: int = 60000
    regime_spread_bins: List[float] = Field(default_factory=lambda: [1.0, 2.5])
    regime_vol_bins: List[float] = Field(default_factory=lambda: [0.0005, 0.0015])
    regime_depth_bins: List[float] = Field(default_factory=lambda: [50.0, 150.0])
    spike_return_threshold: float = 0.0015
    spike_ofi_threshold: float = 5.0
    spike_vol_mult: float = 3.0
    winsor_limits: Dict[str, float] = Field(default_factory=dict)
    anomaly_features: List[str] = Field(default_factory=lambda: ["ofi", "microprice_velocity", "cancel_rate_instant"])
    anomaly_z_limit: float = 9.0
    feature_schema_version: str = "3"
    config_hash: Optional[str] = None


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


class SpreadGateConfig(BaseModel):
    enabled: bool = True
    max_ticks: float = 3.0


class VolatilityGateConfig(BaseModel):
    enabled: bool = True
    quantile: float = 0.9
    lookback_ms: int = 60000
    disable_high_regime: bool = False
    high_quantile: float = 0.9


class LiquidityGateConfig(BaseModel):
    enabled: bool = True
    min_bid_qty: float = 50.0
    min_ask_qty: float = 50.0
    top_levels: int = 3


class AnomalyGateConfig(BaseModel):
    enabled: bool = True
    features: List[str] = Field(default_factory=lambda: ["ofi", "microprice_velocity", "cancel_rate_instant"])
    z_limit: float = 9.0
    require_double_confirmation: bool = True


class SessionWindow(BaseModel):
    start: str
    end: str


class NewsWindow(BaseModel):
    start_ts: int
    end_ts: int


class SessionGateConfig(BaseModel):
    enabled: bool = False
    timezone: str = "UTC"
    open_windows: List[SessionWindow] = Field(default_factory=list)
    tighten_multiplier: float = 1.0
    news_windows: List[NewsWindow] = Field(default_factory=list)
    post_event_cooldown_ms: int = 300_000


class CrossAssetGateConfig(BaseModel):
    enabled: bool = False
    leader_symbol: str = "BTCUSDT"
    max_lag_ms: int = 500
    feature: str = "microprice_velocity"
    direction: str = "same"  # same | opposite
    debounce_count: int = 2
    debounce_window_ms: int = 200


class LeadLagAgreeConfig(BaseModel):
    n: int = 3
    m: int = 4


class LeadLagGateConfig(BaseModel):
    enabled: bool = False
    leader: str = "BTCUSDT"
    agree_required: LeadLagAgreeConfig = Field(default_factory=LeadLagAgreeConfig)


class GatesConfig(BaseModel):
    spread: SpreadGateConfig = Field(default_factory=SpreadGateConfig)
    volatility: VolatilityGateConfig = Field(default_factory=VolatilityGateConfig)
    liquidity: LiquidityGateConfig = Field(default_factory=LiquidityGateConfig)
    cross_asset: CrossAssetGateConfig = Field(default_factory=CrossAssetGateConfig)
    leadlag: LeadLagGateConfig = Field(default_factory=LeadLagGateConfig)
    anomaly: AnomalyGateConfig = Field(default_factory=AnomalyGateConfig)
    session: SessionGateConfig = Field(default_factory=SessionGateConfig)


class BackpressureConfig(BaseModel):
    enabled: bool = True
    stream: str = "sig:candidates"
    group: str = "aiscore"
    pending_threshold: int = 250
    release_threshold: int = 120
    drop_rate: float = 0.5
    check_interval_ms: int = 750


class SignalEngineConfig(BaseModel):
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    gates: GatesConfig = Field(default_factory=GatesConfig)
    tp_pct: float = 0.006
    sl_pct: float = 0.003
    backpressure: BackpressureConfig = Field(default_factory=BackpressureConfig)


class AIScorerWindow(BaseModel):
    frequency_hz: int = 50
    seconds: int = 2
    depth_levels: int = 10


class AIScorerThresholds(BaseModel):
    entry: float = 0.65
    hold: float = 0.45
    precision_target: float = 0.65


class AIScorerCalibration(BaseModel):
    enabled: bool = True
    method: str = "platt"  # platt | isotonic


class AIScorerRolloutGuardConfig(BaseModel):
    winrate_delta_pp: float = 5.0
    min_sample: int = 50
    psi_block: float = 0.30
    ks_block: float = 0.20
    cooldown_minutes: int = 15


class AIScorerAdaptiveConfig(BaseModel):
    enabled: bool = True
    interval_sec: int = 300
    step: float = 0.02
    window: int = 200
    min_avg_pnl: float = 0.0
    safety_precision: float = 0.55
    safety_windows: int = 2
    freeze_reset_precision: float = 0.62
    max_delta: float = 0.15
    min_entry: float = 0.05
    max_entry: float = 0.95
    base_entry: Optional[float] = None


class AIScorerSpreadAdaptiveConfig(BaseModel):
    enabled: bool = True
    spread_regime_adjustments: List[float] = Field(default_factory=lambda: [0.0, 0.02, 0.05])
    volatility_regime_penalty: List[float] = Field(default_factory=lambda: [0.0, 0.01, 0.02])
    clamp: Tuple[float, float] = (0.05, 0.95)


class AIScorerDriftFeatureConfig(BaseModel):
    name: str
    bins: List[float] = Field(default_factory=list)
    psi_alert: float = 0.25
    ks_alert: float = 0.2


class AIScorerDriftConfig(BaseModel):
    enabled: bool = False
    window: int = 500
    emit_interval: int = 100
    features: List[AIScorerDriftFeatureConfig] = Field(default_factory=list)
    score_bins: List[float] = Field(default_factory=lambda: [i / 10 for i in range(11)])
    psi_alert: float = 0.25
    ks_alert: float = 0.2
    brier_alert: float = 0.2
    ece_alert: float = 0.1
    brier_baseline: float = 0.15


class AIScorerShadowConfig(BaseModel):
    enabled: bool = False
    use_threshold_from_meta: bool = True
    tau_entry: Optional[float] = None
    tau_hold: Optional[float] = None
    # Legacy fields retained for backward compatibility
    model_path: Optional[str] = None
    meta_path: Optional[str] = None
    window: int = 500
    promote_delta: float = 0.05
    eval_threshold: float = 0.6


class AIScorerDeploymentConfig(BaseModel):
    mode: str = "active"
    rollback_precision: float = 0.5
    parity_fail_action: str = "fallback"
    cooldown_sec: int = 600
    auto_failover: bool = True


class AIScorerBatchConfig(BaseModel):
    enabled: bool = True
    max_batch: int = 32
    flush_interval_ms: int = 20


class AIScorerConfig(BaseModel):
    # Silence Pydantic warning for field name starting with 'model_'
    model_config = ConfigDict(protected_namespaces=())
    model_type: str = "lightgbm_meta"
    model_path: str = "/app/models/lgbm_meta.onnx"
    meta_path: str = "/app/models/lgbm_meta.json"
    thresholds: AIScorerThresholds = Field(default_factory=AIScorerThresholds)
    batch_size: int = 64
    feature_order: List[str] = Field(default_factory=list)
    calibration: AIScorerCalibration = Field(default_factory=AIScorerCalibration)
    use_meta_thresholds: bool = True
    adaptive: AIScorerAdaptiveConfig = Field(default_factory=AIScorerAdaptiveConfig)
    spread_adaptive: AIScorerSpreadAdaptiveConfig = Field(default_factory=AIScorerSpreadAdaptiveConfig)
    drift: AIScorerDriftConfig = Field(default_factory=AIScorerDriftConfig)
    shadow: AIScorerShadowConfig = Field(default_factory=AIScorerShadowConfig)
    deployment: AIScorerDeploymentConfig = Field(default_factory=AIScorerDeploymentConfig)
    rollout_guard: AIScorerRolloutGuardConfig = Field(default_factory=AIScorerRolloutGuardConfig)
    window: AIScorerWindow = Field(default_factory=AIScorerWindow)
    batch_inference: AIScorerBatchConfig = Field(default_factory=AIScorerBatchConfig)


class LadderConfig(BaseModel):
    fractions: List[float] = Field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    step_ticks: int = 2
    use_percent: bool = False


class ExecutorSizingConfig(BaseModel):
    leverage: float = 10.0
    notional_cap_usd: float = 20000.0
    use_all_in: bool = True
    min_free_balance_usd: float = 0.0
    lot_size_rounding: str = "floor"


class LiquidityBySideConfig(BaseModel):
    enabled: bool = False
    alpha: float = 1.5
    levels: int = 3


class ExecutorGatesConfig(BaseModel):
    liquidity_by_side: LiquidityBySideConfig = Field(default_factory=LiquidityBySideConfig)




class MicropriceFlipGuardConfig(BaseModel):
    enabled: bool = True
    flips: int = 2
    window_ms: int = 400
    cooldown_ms: int = 1500


class FillProbabilityConfig(BaseModel):
    base: float = 0.3
    queue_weight: float = 0.35
    qi_weight: float = 0.25
    depth_weight: float = 0.2
    ladder_decay: float = 0.35
    spread_penalty: float = 0.05
    queue_proxy_cap: float = 1.0
    min_value: float = 0.05
    max_value: float = 0.95


class ExposureCapsConfig(BaseModel):
    max_positions: int = 10
    per_symbol: Dict[str, int] = Field(default_factory=dict)
    max_notional_per_order: float = 0.0
    kelly_fraction: float = 0.05
    reference_equity: float = 1.0


class ExecutorFailSafeConfig(BaseModel):
    enabled: bool = True
    duration_sec: int = 300


class ExecutorConfig(BaseModel):
    ladder: LadderConfig = Field(default_factory=LadderConfig)
    tif: str = "GTC"
    post_only: bool = True
    cancel_timeout_ms: int = 2500
    retry_backoff_ms: List[int] = Field(default_factory=lambda: [200, 400, 800, 1600])
    adaptive_stop_k: float = 2.0
    min_stop: float = 0.002
    qi_aggressive_threshold: float = 0.1
    passive_step_ticks: int = 3
    microprice_guard: bool = True
    max_active_ladders: int = 1
    cooldown_ms: int = 1500
    cancel_latency_p95_ms: int = 1200
    min_fill_probability: float = 0.25
    fill_probability: FillProbabilityConfig = Field(default_factory=FillProbabilityConfig)
    exposure_caps: ExposureCapsConfig = Field(default_factory=ExposureCapsConfig)
    fat_finger_ticks: float = 8.0
    fat_finger_notional: float = 0.0
    sizing: ExecutorSizingConfig = Field(default_factory=ExecutorSizingConfig)
    gates: ExecutorGatesConfig = Field(default_factory=ExecutorGatesConfig)
    microprice_flip_guard: MicropriceFlipGuardConfig = Field(default_factory=MicropriceFlipGuardConfig)
    warmup_seconds: int = 90
    fail_safe: ExecutorFailSafeConfig = Field(default_factory=ExecutorFailSafeConfig)


class MonitorConfig(BaseModel):
    http_host: str = "0.0.0.0"
    http_port: int = 8000


class NearLiqGuardConfig(BaseModel):
    enabled: bool = False
    sl_buffer_multiplier: float = 1.5




class PrecisionHaltConfig(BaseModel):
    window: int = 50
    min_precision: float = 0.55
    safety_precision: float = 0.5
    freeze_step: float = 0.02
    consecutive_windows: int = 2


class MarkoutHaltConfig(BaseModel):
    window: int = 30
    min_markout: float = 0.0


class VolatilityHaltConfig(BaseModel):
    enabled: bool = True
    max_consecutive: int = 3
    precision_floor: float = 0.5
    debounce_ms: int = 60000


class TradeSLOConfig(BaseModel):
    window_seconds: int = 3600
    max_trades: int = 200
    alert_only: bool = False
    p95_prev_release: int = 120


class TradeClusterConfig(BaseModel):
    enabled: bool = False
    window_ms: int = 60000
    max_trades: int = 3
    cumulative_loss: float = -0.0005


class RiskExposureConfig(BaseModel):
    max_positions: int = 10
    per_symbol: Dict[str, int] = Field(default_factory=dict)
    kelly_fraction: float = 0.05
    reference_equity: float = 1.0


class FailSafeConfig(BaseModel):
    enabled: bool = True
    mode: str = "paper"
    duration_sec: int = 300


class BackfillScannerConfig(BaseModel):
    enabled: bool = True
    parquet_path: str = "data/parquet"
    timestamp_column: str = "ts"
    gap_threshold_ms: int = 60_000
    check_interval_sec: int = 300


class RiskConfig(BaseModel):
    enable: bool = True
    precision: PrecisionHaltConfig = Field(default_factory=PrecisionHaltConfig)
    markout: MarkoutHaltConfig = Field(default_factory=MarkoutHaltConfig)
    volatility: VolatilityHaltConfig = Field(default_factory=VolatilityHaltConfig)
    trade_cluster: TradeClusterConfig = Field(default_factory=TradeClusterConfig)
    trade_slo: TradeSLOConfig = Field(default_factory=TradeSLOConfig)
    exposure: RiskExposureConfig = Field(default_factory=RiskExposureConfig)
    near_liq_guard: NearLiqGuardConfig = Field(default_factory=NearLiqGuardConfig)
    assumed_sl_pct: float = 0.0
    fail_safe: FailSafeConfig = Field(default_factory=FailSafeConfig)
    backfill_scanner: BackfillScannerConfig = Field(default_factory=BackfillScannerConfig)


class AppConfig(BaseModel):
    # Back-compat fields (deprecated): prefer market.use and market.symbols
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    symbols: List[str] = Field(default_factory=list)
    redis: RedisConfig
    collector: CollectorConfig = Field(default_factory=CollectorConfig)
    signal_engine: SignalEngineConfig = Field(default_factory=SignalEngineConfig)
    ai_scorer: AIScorerConfig = Field(default_factory=AIScorerConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    rollout_guard: RolloutGuardConfig = Field(default_factory=RolloutGuardConfig)
    # New unified controls
    exchange_mode: str = Field(default="live_paper")  # live_paper | live_real

    class MarketConfig(BaseModel):
        use: str = Field(default="binance_futs")  # binance_futs | bybit_v5 | fake
        symbols: List[str] = Field(default_factory=list)
        binance_ws_public: str = Field(default="wss://fstream.binance.com/stream")
        bybit_ws_public: str = Field(default="wss://stream.bybit.com/v5/public")
        # Bybit v5 category: linear | spot | inverse
        category: str = Field(default="linear")

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


def _feature_config_hash(cfg: FeatureConfig) -> str:
    payload = cfg.model_dump(mode="json", exclude={"config_hash"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_app_config(path: Optional[str] = None) -> AppConfig:
    cfg_path = path or os.environ.get("APP_CONFIG", "config/app.yml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    expanded = _expand_env(raw)
    try:
        cfg = AppConfig.model_validate(expanded)
        # Back-compat: mirror market/use and market/symbols into legacy fields if missing
        if not cfg.exchange.venue:
            cfg.exchange.venue = cfg.market.use
        if cfg.exchange.venue:
            cfg.market.use = cfg.exchange.venue
        if not cfg.symbols:
            cfg.symbols = list(cfg.market.symbols)
        cfg.exchange.mode = (cfg.exchange.mode or "paper").lower()
        cfg.execution.mode = cfg.exchange.mode
        cfg.exchange_mode = "live_real" if cfg.exchange.mode == "real" else "live_paper"
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
            cfg.ai_scorer.thresholds.entry = cfg.signals.score_threshold
            if cfg.ai_scorer.thresholds.hold >= cfg.ai_scorer.thresholds.entry:
                cfg.ai_scorer.thresholds.hold = max(0.1, cfg.signals.score_threshold * 0.7)
            if cfg.ai_scorer.thresholds.precision_target <= 0:
                cfg.ai_scorer.thresholds.precision_target = cfg.signals.score_threshold
            # ladder setup
            cfg.executor.ladder.fractions = cfg.signals.ladder
            cfg.executor.ladder.step_ticks = cfg.signals.ladder_step_ticks
        feature_hash = _feature_config_hash(cfg.signal_engine.features)
        cfg.signal_engine.features.config_hash = feature_hash
        if cfg.ai_scorer.adaptive.base_entry is None:
            cfg.ai_scorer.adaptive.base_entry = cfg.ai_scorer.thresholds.entry
        return cfg
    except ValidationError as e:
        raise SystemExit(f"Invalid configuration {cfg_path}: {e}")
