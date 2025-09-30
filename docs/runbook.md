# Incident Runbook

## Precision Halt
1. Confirm `metrics:risk` precision values in Grafana.
2. Check drift metrics (`feature_psi`, `calibration_expected_calibration_error`).
3. Verify fallback triggered a `ROLLBACK`. If not, issue `control:events` MODE_SWITCH to paper mode.
4. Inspect latest model artifacts (reliability, threshold sweep) and consider redeploy of previous version.

## Volatility Halt
1. Review market regime (spread/depth) and news calendars.
2. Confirm trade SLO and exposure metrics remain within bounds.
3. Resume only after volatility counter resets and `trade_rate` normalizes.

## Parity Fail
1. Inspect `onnx_parity.log` artifact and CI logs.
2. Ensure train-time parity metric triggered rollback; system should be in paper mode.
3. Rebuild ONNX from latest metadata; rerun canary pipeline before returning to live.

## Rollback Guard Trigger
1. Review the `MODE_SWITCH` control event reason (`winrate_delta`, `onnx_parity_fail`, `psi_feature_*`, `ks_feature_*`).
2. Inspect `winrate_baseline_active`, `winrate_canary`, and drift metrics in Grafana to confirm the breach.
3. Verify `models/registry/active/model.json` now points to the previous stable model from `manifest.json`.
4. Leave rollout disabled during the cooldown window; once metrics stabilise, promote a fixed model and re-enable canary mode.

## Data Gaps / Quality Issues
1. Run `scripts/scan_parquet_gaps.py --root data/parquet`.
2. Validate exchange reference using `scripts/validate_symbols.py`.
3. If gaps detected, trigger backfill job and re-train before enabling trading.
