# RL Scalper Workflow

## 1. Collect Historical Data
- Configure symbol, start/end dates, depth levels, on-chain feeds.
- Run `python scripts/download_rl_dataset.py --symbol BTCUSDT --start 2023-01-01T00:00:00 --end 2024-01-01T00:00:00 --raw-dir data/raw --processed-dir data/processed`.
- Script downloads Binance depth archives, spot trades, funding rates, optional on-chain metrics, and produces train/val/test `.npz` splits under `data/processed`.

## 2. Train the Agent
- Launch training with `python -m app.rl.scalper_train --dataset data/processed/train.npz --episodes 300 --device cuda`.
- Checkpoints and TensorBoard logs land in `runs/scalper/`.
- Optional HPO: add `--optuna-trials 20` to run Optuna sweeps (requires `optuna` package).

## 3. Evaluate & Benchmark
- Evaluate on hold-out split: `python scripts/evaluate_scalper.py --dataset data/processed/test.npz --model runs/scalper/best_model.pt`.
- Script reports Sharpe/Sortino/max drawdown vs buy-and-hold and random policies.
- For additional baselines, adapt `scripts/evaluate_scalper.py` to plug other agents (PPO/SAC) using the same environment.

## 4. Risk Controls Integration
- `app/rl/risk_manager.py` defines leverage caps, liquidation buffer checks, and funding adjustments. Tune `RiskParams` to reflect venue constraints.
- Environment surfaces realized volatility and funding rates so the agent can adjust exposure dynamically.
- Daily loss caps and drawdown measurements flow into rewards via the terminal Sharpe/Sortino bonus.

## 5. Deployment Checklist
- Export Torch weights (`best_model.pt`) and accompanying config (price/size actions, risk params).
- Register evaluation metrics and data windows in `models/registry/manifest.json`.
- For shadow / canary runs reuse `scripts/canary_pipeline.py` with the new policy plugged into the execution stack.
