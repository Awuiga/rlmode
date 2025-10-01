# Incident Runbook

## Precision Halt
1. Confirm the alerting signal by checking Grafana panels for `precision`, `winrate_rolling{route}` and `winrate_baseline_active`.
2. Inspect drift (`psi_feature_*`, `ks_feature_*`), calibration (`ece_online`, `brier_online`) and canary sample size to identify if degradation is systemic.
3. If automatic rollback did **not** trigger, immediately issue a manual rollback (see [Operational Commands](#operational-commands)). Keep rollout disabled until post-mortem completes.
4. Compare latest model reliability plots, threshold sweep and feature deltas against the previous release; restore the last good manifest entry if regression is confirmed.

## Volatility Halt
1. Review depth/spread metrics and market news calendars to confirm the halt is expected (e.g. CPI, FOMC, liquidation cascades).
2. Validate that execution SLOs (`avg_markout_*`, `fill_rate_rolling`, `cancel_to_fill_ratio`) remain within acceptable tolerances.
3. Resume trading only when volatility monitors clear, the warm-up timer elapses, and `trades_per_hour` returns to pre-event levels.

## Parity Fail
1. Inspect `onnx_parity.log` and CI artifacts for operator error (mismatched feature order, calibration payload).
2. Verify the guard rails switched rollout to `disabled` and incremented `rollback_total{reason="onnx_parity_fail"}`.
3. Re-export the ONNX from the training artifacts, re-run `scripts/canary_pipeline.py`, and redeploy only after the parity metric stays at zero in staging.

## Drift Breach
1. For PSI/KS alerts inspect the Grafana Drift panel to isolate offending features and the time window.
2. Cross-check data sources (collector stats, parquet backfill scanner) for schema or feed anomalies.
3. If drift persists or impacts win rate, promote the previous stable model, disable rollout, and trigger a data audit before resuming the release.

## Throughput Spike
1. Confirm the alert `trades_per_hour > p95_prev_release * 1.5` is genuine by checking Redis queue depth, backpressure counters, and `exec_ladder_aggressiveness_total`.
2. Enable temporary backpressure by tightening gates or reducing `rollout.canary_fraction`; consider switching to paper if fill rate degrades.
3. After volume normalises, review latency profiles and cancel ratios before restoring the original rollout settings.


## Warm-up Gate Active
1. Warm-up drops appear as `warmup_drop_total{symbol}` spikes when the executor restarts or a service is recycled.
2. Confirm book depth and spreads look healthy before shortening `warmup.seconds`; never bypass the gate if collector lag or replay is ongoing.
3. Wait for the configured period or restart once more with a higher `warmup.seconds` value if repeated restarts occur during volatile market opens.

## Zero Liquidity / Stale Book
1. Inspect market depth snapshots and exchange websockets for disconnects; if liquidity is zero on either side, halt trading immediately.
2. Confirm `liquidity_by_side` and `near_liq_guard` gates are firing; review `warmup_drop_total` for symbols entering protective mode.

3. Switch the executor to paper (or stop trading) until order book depth and spreads recover for a sustained interval (at least 5 minutes of normalised depth).

## Rollback Guard Trigger
1. Review the emitted `MODE_SWITCH{from,to}` control event and `rollback_total` labels to understand the trigger (`winrate_delta`, `onnx_parity_fail`, `psi_feature_*`, `ks_feature_*`).
2. Validate that `models/registry/active/model.json` now points to the prior manifest entry and that cooldown is active.
3. Use the commands below to recover once metrics stabilise (typically after the 15 minute cooldown).

## Fail-safe Alert

1. `control:events` with `severity="CRIT"` forces the executor into paper mode for the configured cooldown; inspect `fail_safe_trigger_total{reason}` and the accompanying `mode_switch_total{from,to}` labels for the trigger.
2. Ensure `warmup_drop_total` and `backpressure_total` remain flat while in paper mode; once `fail_safe_trigger_total` stops incrementing, the cooldown has elapsed.
3. Diagnose the upstream cause (connectivity, model outputs, exchange API errors) before restoring real execution with `EXCHANGE_MODE=real docker compose --profile real up -d ai_scorer executor risk`.

## Backpressure / Queue Build-up
1. Monitor `backpressure_total{mode}` and Redis stream depth (`XLEN sig:approved`) when alerts fire.
2. Allow the automatic throttle to drain the queue; if backlog persists, raise gate thresholds or set `backpressure.drop_mode="halt"` temporarily to shed load.
3. Once the queue returns below `backpressure.max_queue_len`, confirm canary win rate and fill rate recover before re-enabling full traffic.


## Data Gaps / Quality Issues
1. Run `scripts/scan_parquet_gaps.py --root data/parquet` to locate missing partitions.
2. Validate exchange references with `PYTHONPATH=. python scripts/validate_symbols.py --symbols data/reference/symbols.yml`.
3. If gaps are detected, trigger backfill jobs, retrain, and repeat parity plus canary validation before re-enabling rollout.

## Operational Commands
- **Rollback to previous active model**: `PYTHONPATH=. python scripts/promote_model.py --to active --id <PREVIOUS_MODEL_ID>` followed by `bash scripts/release.sh --shadow-id <PREVIOUS_MODEL_ID> --disable-rollout` if you need to keep rollout off.
- **Promote candidate to shadow**: `PYTHONPATH=. python scripts/promote_model.py --to shadow --id <MODEL_ID>`.
- **Promote shadow to active**: `PYTHONPATH=. python scripts/promote_model.py --to active --id <MODEL_ID>` once canary criteria are satisfied.
- **Switch execution to paper immediately**: set `exchange.mode="paper"` in `config/app.yml` (or export `EXCHANGE_MODE=paper` before restarting the executor) and restart the stack: `EXCHANGE_MODE=paper docker compose --profile paper up -d ai_scorer executor`.
