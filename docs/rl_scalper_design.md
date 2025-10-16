# RL Scalper Architecture Overview

## Objectives
- Build a risk-aware high-frequency RL scalper for crypto perpetual futures.
- Consume one year of spot and level-2 limit order book (LOB) history enriched with funding and on-chain factors.
- Train, validate, and evaluate an agent that controls limit order placement (price/size) under leverage and liquidation constraints.

## Data Pipeline
- **Sources**
  - Spot trades and L2 book snapshots (Binance default; pluggable exchanges via `app.exchange`).
  - Funding rates from perpetual futures endpoints.
  - On-chain metrics via configurable HTTP APIs (e.g., CryptoQuant / Glassnode; wrapped in interface for custom providers).
- **Fetcher Layer (`app/rl/data/fetchers`)**
  - Async REST clients with cursor/pagination handling and throttling.
  - Converts raw API responses into normalized records (`TradeTick`, `OrderBookLevel`, `FundingRate`, `OnChainMetric` dataclasses).
- **Orchestration (`HistoricalDataCollector`)**
  - Drives multi-source downloads, persists to Parquet under `data/raw/<symbol>/YYYY/MM/DD`.
  - Computes derived book features: spread, micro/macro depths, imbalance, OFI, realized volatility.
  - Aligns funding/on-chain series to tick timeline using forward-fill with latency guards.
- **Regime Splitter (`MarketRegimeSplitter`)**
  - Segments history into bull/bear/range using rolling return and volatility thresholds.
  - Produces train/validation/test manifests ensuring coverage of each regime.
- **Feature Materialization**
  - Invokes `app.features.compute_features_offline` for consistency with production stack.
  - Stores model-ready tensors in `data/processed/<split>/<symbol>/*.npz`.

## Simulation Environment
- **ABIDES Integration (`app/rl/abides_env.py`)**
  - Wraps ABIDES market simulator when installed; falls back to internal discrete-event engine otherwise.
  - Replays historical order flow and injects agent limit orders into the matching engine with queue priority handling.
  - Models execution latency, network jitter, and stochastic slippage; configurable maker/taker fees.
- **Observation Builder**
  - Stacks: current L2 snapshot (top 20 levels), engineered book features, short-term realized volatility, funding rates, on-chain indicators, and agent inventory stats.
  - Supports windowed micro features (per-second sequence) and macro aggregates (per-minute/hour bars).
- **Action Space**
  - Branch 1: discrete price offsets (inside spread, at-touch, passive levels).
  - Branch 2: discrete order sizes (risk-scaled).
  - Auxiliary head predicts conditional variance to drive adaptive leverage.
- **Reward Function (`risk_reward`)**
  - Step reward = ΔPnL − trade_penalty − liquidity_penalty.
  - Episode bonus mixes daily PnL, Sharpe/Sortino proxies, and drawdown penalties.

## Agent & Modeling
- **Encoder (`MultiStreamStateEncoder`)**
  - LSTM over micro tick window + MLP over macro indicators -> fused latent.
- **Network (`RiskAwareBranchingDQN`)**
  - Shared extractor → dueling branching heads (price/size) + risk head.
  - Optional distributional critic and quantile regression toggle.
- **Experience Handling**
  - Prioritized replay with stratified sampling across market regimes.
  - N-step returns and Double DQN target updates.
- **Alternatives**
  - PPO/SAC variants share encoder; action projection adapts to continuous policies (implemented in modular policy factory).

## Risk Management
- **Dynamic Leverage Controller**
  - Uses risk head volatility estimate + realized volatility to set leverage cap per step.
  - Enforces margin requirements, liquidation buffer, and funding-aware carry cost.
- **Position Limits**
  - Hard cap on notional exposures; soft penalties for repeated breaches.
- **Kill Switches**
  - Daily loss cap, drawdown guard, loss streak halt integrated with existing risk routes (`app/risk`).

## Training, Validation, Evaluation
- **Training Loop (`app/rl/scalper_train.py`)**
  - Curriculum from low-latency simulated episodes to full ABIDES replay.
  - Mixed precision support, checkpointing, TensorBoard logging.
- **Hyperparameter Optimisation**
  - Optuna study orchestrated via `scripts/hpo_scalper.py`; objective aggregates Sharpe, Sortino, PnL stability.
  - Trials pull from regime-aware validation windows.
- **Evaluation Suite**
  - Benchmarks: buy-and-hold, TWAP, existing RL baselines.
  - Reports PnL, Sharpe/Sortino, max drawdown, trade stats, liquidation incidents.
  - Generates notebooks under `docs/notebooks/` for qualitative review.

## Deployment Hooks
- Model registry entry with metadata (data ranges, metrics, risk parameters).
- Export TorchScript/ONNX policy + encoder; package risk/leverage config for live executor.
- Integration tests simulate shadow trading using `scripts/canary_pipeline.py`.

