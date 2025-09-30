# Model Card

- **Model**: LightGBM Meta Scorer
- **Source data**: See `models/latest/lgbm_meta.json[data]`
- **Training timestamp**: populate from metadata `generated_ts`
- **Key metrics**: PR-AUC, precision@k, calibration (reliability.json, pr_curve.json)
- **Feature schema**: Stored in `feature_order.json`
- **Drift baselines**: See `monitoring` section inside metadata for score/feature bins.
- **Known limitations**:
  - Assumes Bybit linear futures microstructure features.
  - Maker-only execution; taker costs not modelled.
- **Operational checks**:
  - `onnx_parity.log` reports parity status.
  - Canary pipeline must pass SLO thresholds before deploy.
  - Shadow model metrics (`shadow_average_precision_delta`) monitored for regressions.

## Registry Metadata
- `model.json` must include `id`, `schema_hash`, `calibration`, and `thresholds` (`tau_entry`, `tau_hold`).
- `schema_hash` must match the FeatureState hash at runtime; mismatches disable deployment.
- `thresholds` are used for both live gating and shadow evaluation.
