

### Development

- All services must be launched via `python -m app.<service>.main`.
- Local artifacts (.venv, __pycache__, data/redis, .env) must not be committed.

#### Risk controls

- `daily_loss_cap_usd` halts risk routing for the day once realised PnL breaches the configured negative threshold and emits `daily_cap_trigger_total` with drop reason `daily_cap`.
- `loss_streak_max` tracks consecutive losing trades; once the streak length reaches the limit, risk stops taking signals, records `streak_halt_total`, and applies `session_cooldown_minutes` before trading resumes.
- `near_liquidation_short` protects short entries by comparing the estimated liquidation price to the assumed stop-loss distance and drops signals with reason `near_liquidation_short` if collateral is too thin.

Model pipeline recipe
- regenerate parquet with `python dump_market_to_parquet.py --config config/app.yml`
- apply labels via `python labeling.py` or the ETL job defined in `labeling.py`
- train scorer with `python train.py --data data/training.parquet --output-dir models/latest`
- export ONNX and metadata from the training step (stored in the output directory)
- simulate using `scripts/canary_pipeline.py` across normal/event/high-vol days
- deploy by copying the ONNX and metadata into `models/` and restarting `ai_scorer`

### RL Scalper pipeline

- Collect historical spot + L2 order book data, funding, and on-chain metrics with `python scripts/download_rl_dataset.py --symbol BTCUSDT --start 2023-01-01T00:00:00 --end 2024-01-01T00:00:00`.
- Train the risk-aware branching DQN agent via `python -m app.rl.scalper_train --dataset data/processed/train.npz --episodes 300 --device cuda`.
- Evaluate vs baselines: `python scripts/evaluate_scalper.py --dataset data/processed/test.npz --model runs/scalper/best_model.pt`.
- Detailed architecture and workflow notes: `docs/rl_scalper_design.md`, `docs/rl_scalper_workflow.md`.

Release checklist
- feature schema hash matches between parquet sidecars and the exported ONNX metadata
- PR-AUC and precision targets meet or beat the current benchmark in `models/latest/lgbm_meta.json`
- ONNX parity passes locally and no `onnx_parity_fail_total` increments in Prometheus
- Canary pipeline passes all gates (winrate >= target, avg markout >= 0, fill-rate >= minimum)

Release policy
- Freeze the deployed thresholds (`tau_entry`, `tau_hold`) for 24 hours after a ramp before enabling auto-adaptive tuning with small step sizes.
- Follow blue/green stages: keep the candidate in shadow for at least one trading day, run canary at 5–10% for 2–4 hours, then ramp through 25% → 50% → 100% while monitoring guard rails.
- Log each release in `models/registry/manifest.json` with benchmark metrics, evaluation windows, and notes about the market regimes covered by the data.
- Use `scripts/release.sh --shadow-id <MODEL_ID>` to orchestrate the validation → promote shadow → enable canary → monitor → promote to full rollout flow. Pass `--disable-rollout` to end the run with rollout disabled after validation or `--skip-validations` when rerunning during the same release.

Advanced toolkit
- `scripts/calibrate_fill_model.py` builds empirical fill curves from historical fills.
- `scripts/simulate_execution.py` runs Monte-Carlo latency and shock scenarios (outputs JSON report).
- `scripts/hpo_optuna.py` executes a precision-first Optuna sweep and exports top trials.
- `scripts/threshold_builder.py` derives regime-specific thresholds using existing calibration sweeps.
- `scripts/scan_parquet_gaps.py` reports missing partitions / gaps in parquet archives.
- `scripts/validate_symbols.py` cross-checks symbol parameters against reference maker/taker fees.
- `scripts/validate_config.py` ensures `config/app.yml` satisfies Pydantic schema.
- `scripts/profile_service.py` runs cProfile on any entrypoint for latency budgeting.
- `app/rl/daily_env.py` hosts a Gymnasium-compatible environment that optimises daily PnL with hindsight bonuses and trade-count penalties.
- `app/rl/branching_dqn.py` provides a dueling branching DQN (price + size heads) ready to plug into the RL training loop.
- `app/rl/reward_fn.py` exposes `calculate_reward` for daily PnL shaped rewards in Gym environments.
- `app/rl/metrics.py` implements `calculate_metrics` for Sharpe/Sortino/drawdown/winrate reporting during training.
- `app/rl/multi_stream_encoder.py` merges tick-level LSTM features with macro indicators into a unified state embedding.
- `app/rl/risk_branching_dqn.py` extends the branching DQN with a volatility head and combined training loss.
- `app/rl/trading_env.py` implements a Gymnasium trading simulator with fees, slippage, trade caps and capital-based rewards.
- `app/rl/train_loop.py` contains a full DQN training loop with checkpoints, TensorBoard logging, and early stopping.
- `logger.py` is a standalone utility that streams metrics to TensorBoard/CSV and produces PDF/PNG reports.

Operational docs
- `docs/model_card.md` summarises dataset, metrics and assumptions for the latest model.
- `docs/runbook.md` documents response steps for precision halts, volatility halts and parity failures.

Exchange modes
- Configure runtime venue in `config/app.yml`:

```yaml
exchange:
  mode: ${EXCHANGE_MODE:-paper}
  venue: ${EXCHANGE_VENUE:-bybit_v5}
  maker_only: true
  reduce_only: true
  post_only: true
```

- CLI (local):

```bash
EXCHANGE_MODE=paper python -m app.executor.main
EXCHANGE_MODE=real python -m app.executor.main
```

- Docker Compose profiles:

```bash
docker compose --profile paper up collector signal_engine ai_scorer executor
EXCHANGE_MODE=real docker compose --profile real up collector signal_engine ai_scorer executor
```

> **Important:** Services must be invoked via `python -m app.<service>.main`. The root-level scripts remain as thin compatibility wrappers and should not be used for deployments or automation.


Canary & Rollback
- Enable deterministic hashing by updating `config/app.yml`:

```yaml
rollout:
  mode: "canary"      # disabled | canary | full
  canary_fraction: 0.10
  rng_seed: 1337
```

- Start the executor against a real venue so that ~10% of approved signals hash to `route="canary"` (real exchange) and the remainder stay on the paper simulator: `EXCHANGE_MODE=real python -m app.executor.main`.
- The executor tags every order with `trades_opened_total{route="canary|paper|full"}` so downstream dashboards can split performance. Risk publishes per-route gauges for `trades_won_total`, `winrate_rolling`, `fill_rate_rolling`, `avg_markout_{100,500,1000}ms`, and `cancel_to_fill_ratio`. Canary supervision exports `winrate_baseline_active`, `winrate_canary`, `sample_canary`, `mode_switch_total`, `rollback_total`, `brier_online`, `ece_online`, `brier_baseline_active`, `psi_feature_*`, `psi_feature{feature}`, `ks_feature_*`, `ks_feature{feature}`, `tau_entry_current{model}`, `tau_hold_current{model}`, `trades_per_hour`, and `trades_per_hour_prev_release_p95`. AI scoring exposes `model_score_distribution{split="online"}`, `shadow_score_distribution{split="online"}`, `onnx_parity_fail_total`, and `parquet_schema_version_mismatch_total` while the executor tracks `exec_ladder_aggressiveness_total{bucket}`.
- Guard rails live under `rollout_guard` in `config/app.yml` (winrate delta in percentage points, minimum sample, PSI/KS blocks, cooldown window). Canary is automatically rolled back when **any** of the following hold:
  1. `sample_canary >= min_sample` **and** `winrate_canary < winrate_baseline_active - winrate_delta_pp`
  2. `increase(onnx_parity_fail_total[10m]) > 0`
  3. `max_over_time(psi_feature_*[30m]) > psi_block` **or** `max_over_time(ks_feature_*[30m]) > ks_block`
- On rollback the supervisor switches `models/registry/active` back to the previous manifest entry, writes `rollout.mode="disabled"`, emits a `MODE_SWITCH{from,to}` control event, and increments `mode_switch_total`/`rollback_total` with the reason label. Cooldown (default 15 minutes) prevents thrashing while operators investigate.
- Promote a new build or revert manually with `scripts/promote_model.py --to active --id MODEL_ID` (forward) and `scripts/promote_model.py --to active --id PREVIOUS_ID` (rollback). Re-enable full routing by setting `rollout.mode="full"`; set `"disabled"` to fall back to paper-only execution.

### Observability & Alerts

#### Canary fraction realized
- The executor exports `executor_canary_fraction_realized`, the ratio of canary orders opened to total live (canary + paper) orders opened.
- Alert `CanaryFractionDeviation` fires when the realised fraction diverges from the rollout target by more than three percentage points for 15 minutes; tune the target via `config/rollout.canary_fraction`.

#### Grafana dashboard
- **Win Rate by Route** &mdash; compares `winrate_rolling{route}` against the aggregate `winrate_baseline_active`. Expect canary to track the baseline; drops below the yellow 50% threshold require investigation.
- **Markout & Fill Rate** &mdash; overlays `avg_markout_1000ms{route}` with `fill_rate_rolling{route}`. Markouts should stay above zero while fill rate remains above the 25% green target.
- **Score Distribution (Online)** &mdash; tracks `model_score_distribution{stat="mean"}` and `shadow_score_distribution`. Divergence or collapsing standard deviation warns of calibration or shadow drift.
- **Calibration** &mdash; plots `brier_online`, `brier_baseline_active`, and `ece_online`. The Brier curve should remain below the baseline, ECE below 5%.
- **Drift PSI / KS** &mdash; visualises `psi_feature` and `ks_feature` grouped by feature name. PSI > 0.3 or KS > 0.2 highlights meaningful drift.
- **Execution Ladder Aggressiveness** &mdash; compares `rate(exec_ladder_aggressiveness_total{bucket}[5m])` across aggression buckets to spot skew towards marketable orders.
- **Parity & Schema** &mdash; `increase(onnx_parity_fail_total[5m])` and `increase(parquet_schema_version_mismatch_total[5m])` surface inference or data compatibility issues.
- **Rollouts & Mode Switches** &mdash; shows `increase(mode_switch_total[30m])` and `increase(rollback_total[30m])` to summarise automated and manual interventions.

#### Alert catalogue
- `WinRateDrop` (**critical**) &mdash; `rolling_precision < 0.5` for 10 minutes.
- `CanaryWinrateDrop` (**warning**) &mdash; Canary win rate trails baseline by more than five percentage points over 30 minutes.
- `OnnxParityFailure` (**warning**) &mdash; any increment of `onnx_parity_fail_total` within 10 minutes.
- `DriftDetected` (**critical**) &mdash; `psi_feature` > 0.30 or `ks_feature` > 0.20 in the last 30 minutes.
- `CalibrationOutOfSpec` (**warning**) &mdash; `ece_online > 0.05` or `brier_online` above the live baseline.
- `ThroughputSpike` (**warning**) &mdash; `trades_per_hour` exceeds `trades_per_hour_prev_release_p95` by 50%.
- `ExecutionDegradation` (**critical**) &mdash; `fill_rate_rolling` falls below 25% while high-bucket ladder usage increases.


### Resilience

- **Warm-up gate** &mdash; configure `warmup.enabled`/`warmup.seconds` in `config/app.yml` to delay execution until book state stabilises. Dropped signals increment `warmup_drop_total{symbol}` so dashboards surface cold starts.
- **Backpressure controls** &mdash; `backpressure.enabled`, `max_queue_len`, and `drop_mode` govern how the executor reacts once `sig:approved` queues exceed capacity. In `halt` mode signals are dropped with `backpressure_total{mode="halt"}`, while `degrade` tightens ladder filters and reports `backpressure_total{mode="degrade"}` until depth recovers.
- **Fail-safe mode** &mdash; `fail_safe.switch_to_paper_on_crit` and `cooldown_minutes` trigger an automatic exchange switch when `control:events` delivers a `CRIT` alert. Metrics `fail_safe_trigger_total{reason}` and `mode_switch_total{from,to}` document activations and the subsequent restoration to the baseline venue.

- **Data health** &mdash; the risk service runs a parquet backfill scan (`parquet_gap_detected_ms`) and exchange reference validator (`exchange_reference_mismatch_total`) so operators can catch schema drift or venue setting mismatches before they cascade into production issues.



Sizing & Near-Liq Guard
- Order size uses `compute_order_qty` with leverage 10x and a notional cap of 20k USD.
- Formula: `qty = floor((min(cap, free_balance * leverage) / price) / lot_step) * lot_step`.
  Example: BTC at 60k -> qty ~= floor((min(20000, free_balance*10)/60000)/0.001)*0.001 ~= 0.333.
- Signals are dropped instead of inflating quantity to satisfy exchange minimums.
- Near-liquidation guard blocks longs when `(entry - liq)/entry` is below 1.5 * the assumed stop-loss distance.
- Configure in `config/app.yml`:

```yaml
executor:
  sizing:
    leverage: 10
    notional_cap_usd: 20000
    use_all_in: true
    min_free_balance_usd: 200
    lot_size_rounding: "floor"
  gates:
    liquidity_by_side:
      enabled: true
      alpha: 1.5
      levels: 3
risk:
  near_liq_guard:
    enabled: true
    sl_buffer_multiplier: 1.5
  assumed_sl_pct: 0.003
```

Local checks:
```bash
pytest -q tests/test_sizing.py
pytest -q tests/test_executor_gates.py
```

Runtime:
```bash
EXCHANGE_MODE=paper python -m app.executor.main
docker compose --profile paper up collector signal_engine ai_scorer executor
```

Model Registry & Shadow Scoring
- Place each trained model under `models/registry/<model_id>/` with `model.onnx`, `model.json`, and optional `artifacts/`.
- Update the manifest (`models/registry/manifest.json`) with metadata `{id, created_at, schema_hash, thresholds, path, notes}`.
- Promote slots via `scripts/promote_model.py --to active --id MODEL_ID` or `--to shadow`.
- Enable shadow scoring through `config/app.yml`:

```yaml
ai_scorer:
  shadow:
    enabled: true
    use_threshold_from_meta: true
    tau_entry: null
    tau_hold: null
```

- Metrics exposed: `model_active_id`, `model_shadow_id`, `shadow_signals_{evaluated,approved}_total`, `shadow_score_distribution{split="online"}`, and `tau_entry_current{model}`.
- Toggle shadow off by setting `ai_scorer.shadow.enabled=false` or removing the registry `shadow/` link.
