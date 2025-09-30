

Model pipeline recipe
- regenerate parquet with `python dump_market_to_parquet.py --config config/app.yml`
- apply labels via `python labeling.py` or the ETL job defined in `labeling.py`
- train scorer with `python train.py --data data/training.parquet --output-dir models/latest`
- export ONNX and metadata from the training step (stored in the output directory)
- simulate using `scripts/canary_pipeline.py` across normal/event/high-vol days
- deploy by copying the ONNX and metadata into `models/` and restarting `ai_scorer`

Release checklist
- feature schema hash matches between parquet sidecars and the exported ONNX metadata
- PR-AUC and precision targets meet or beat the current benchmark in `models/latest/lgbm_meta.json`
- ONNX parity passes locally and no `onnx_parity_fail_total` increments in Prometheus
- Canary pipeline passes all gates (winrate >= target, avg markout >= 0, fill-rate >= minimum)

Advanced toolkit
- `scripts/calibrate_fill_model.py` builds empirical fill curves from historical fills.
- `scripts/simulate_execution.py` runs Monte-Carlo latency and shock scenarios (outputs JSON report).
- `scripts/hpo_optuna.py` executes a precision-first Optuna sweep and exports top trials.
- `scripts/threshold_builder.py` derives regime-specific thresholds using existing calibration sweeps.
- `scripts/scan_parquet_gaps.py` reports missing partitions / gaps in parquet archives.
- `scripts/validate_symbols.py` cross-checks symbol parameters against reference maker/taker fees.
- `scripts/validate_config.py` ensures `config/app.yml` satisfies Pydantic schema.
- `scripts/profile_service.py` runs cProfile on any entrypoint for latency budgeting.

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


Canary routing
- Toggle rollout knobs in `config/app.yml`:

```yaml
rollout:
  mode: "canary"
  canary_fraction: 0.10
  rng_seed: 1337
```

- Launch executor against real venue to activate deterministic hashing:

```bash
EXCHANGE_MODE=real python -m app.executor.main
```

- Switch `rollout.mode` to `full` for 100% real flow or back to `disabled` to restore legacy routing.


Online Benchmarking
- The risk supervisor keeps a rolling baseline winrate using `risk.precision.window` samples from active real trades (routes `real` and `full`).
- Canary trades (`route="canary"`) maintain their own rolling winrate and sample counter so you can compare uplift without triggering automatic halts.
- Exported metrics:
  - `winrate_baseline_active`
  - `winrate_canary`
  - `sample_canary`
- When no canary trades are in the window (`sample_canary = 0`), `winrate_canary` is reported as `0.0` by design; treat that as "insufficient data".
- Use Grafana/Prometheus to overlay `winrate_canary` vs `winrate_baseline_active` over identical windows and monitor convergence before promotions.

Rollback
- Configure guard rails via `rollout_guard` in `config/app.yml` (winrate delta, PSI/KS limits, cooldown).
- Canary underperformance beyond the configured delta with sufficient samples triggers an automatic rollback.
- Any increase in `onnx_parity_fail_total` or PSI/KS drift metric breaching the guard thresholds also triggers.
- On rollback the supervisor reverts `models/registry/active` to the previous manifest entry, disables rollout, and emits `MODE_SWITCH` plus `mode_switch_total`/`rollback_total` metrics.
- The cooldown window (default 15 minutes) blocks repeated rollbacks until operators clear the issue.


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
