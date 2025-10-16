"""Trading environment backed by historical LOB replay with ABIDES integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from .risk_manager import RiskManager, RiskParams
from .metrics import calculate_metrics


try:  # pragma: no cover - optional dependency
    import abides_markets  # noqa: F401
    ABIDES_AVAILABLE = True
except ImportError:  # pragma: no cover
    ABIDES_AVAILABLE = False


@dataclass(slots=True)
class ABIDESTradingEnvConfig:
    maker_fee_rate: float = 0.0002
    taker_fee_rate: float = 0.0005
    slippage_bps: float = 1.0
    latency_steps: int = 1
    episode_length: int = 1800
    base_order_size: float = 0.01
    price_tick: float = 0.5
    price_offsets: Sequence[int] = (-2, -1, 0, 1, 2)
    size_levels: Sequence[float] = (0.0, 0.25, 0.5, 1.0)
    initial_equity: float = 100_000.0
    seed: Optional[int] = None


class RewardTracker:
    """Track intraday PnL and compute terminal Sharpe/Sortino bonuses."""

    def __init__(self, trade_penalty: float = 0.01, drawdown_penalty: float = 0.1) -> None:
        self.trade_penalty = trade_penalty
        self.drawdown_penalty = drawdown_penalty
        self.reset(0.0)

    def reset(self, equity: float) -> None:
        self.equity_start = equity
        self.equity_curve = [equity]
        self.trade_count = 0
        self.max_equity = equity
        self.min_equity = equity

    def register_trade(self) -> None:
        self.trade_count += 1

    def step(self, equity: float) -> None:
        self.equity_curve.append(equity)
        self.max_equity = max(self.max_equity, equity)
        self.min_equity = min(self.min_equity, equity)

    def terminal_reward(self) -> Tuple[float, Dict[str, float]]:
        metrics = calculate_metrics(self.equity_curve, episode=1, print_every=0)
        pnl = self.equity_curve[-1] - self.equity_start
        reward = pnl
        reward += 0.5 * metrics["sharpe"] + 0.3 * metrics["sortino"]
        reward -= self.drawdown_penalty * abs(metrics["max_drawdown"])
        reward -= self.trade_penalty * float(self.trade_count)
        return reward, metrics


class ABIDESTradingEnv(gym.Env):
    """Environment that replays collected LOB data with queue-aware execution."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        micro: np.ndarray,
        macro: np.ndarray,
        meta: np.ndarray,
        timestamps: np.ndarray,
        config: Optional[ABIDESTradingEnvConfig] = None,
        risk_manager: Optional[RiskManager] = None,
    ) -> None:
        super().__init__()
        if micro.shape[0] != macro.shape[0] or micro.shape[0] != meta.shape[0]:
            raise ValueError("micro, macro and meta arrays must share the same length")
        self.cfg = config or ABIDESTradingEnvConfig()
        self.micro = micro
        self.macro = macro
        self.meta = meta
        self.timestamps = timestamps
        self.rng = np.random.default_rng(self.cfg.seed)
        self.risk = risk_manager or RiskManager(RiskParams())
        self.reward_tracker = RewardTracker()

        micro_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=micro.shape[1:],
            dtype=np.float32,
        )
        macro_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(macro.shape[1],),
            dtype=np.float32,
        )
        agent_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "micro": micro_space,
                "macro": macro_space,
                "agent": agent_space,
            }
        )
        self.action_space = gym.spaces.MultiDiscrete(
            [len(self.cfg.price_offsets), len(self.cfg.size_levels)]
        )

        self._episode_start = 0
        self._episode_end = 0
        self._idx = 0
        self._position = 0.0
        self._cash = self.cfg.initial_equity
        self._equity = self.cfg.initial_equity
        self._prev_equity = self.cfg.initial_equity
        self._last_price = 0.0
        self._realized_pnl = 0.0
        self._trades_executed = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        num_steps = self.micro.shape[0]
        max_start = max(1, num_steps - self.cfg.episode_length - 1)
        self._episode_start = self.rng.integers(0, max_start)
        self._episode_end = min(self._episode_start + self.cfg.episode_length, num_steps - 1)
        self._idx = self._episode_start
        self._position = 0.0
        self._cash = self.cfg.initial_equity
        self._equity = self.cfg.initial_equity
        self._prev_equity = self.cfg.initial_equity
        self._last_price = float(self.meta[self._idx, 0])
        self._realized_pnl = 0.0
        self._trades_executed = 0
        self.reward_tracker.reset(self._equity)
        self.risk.reset(self._equity)
        obs = self._build_observation()
        info = {"equity": self._equity, "position": self._position}
        return obs, info

    def step(self, action: np.ndarray):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")

        price_offset_idx, size_idx = int(action[0]), int(action[1])
        size_multiplier = self.cfg.size_levels[size_idx]
        trade_side = np.sign(size_multiplier)
        executed = False
        trade_pnl = 0.0

        current_meta = self.meta[self._idx]
        price = float(current_meta[0])
        spread = float(current_meta[1])
        depth_bid = float(current_meta[2])
        depth_ask = float(current_meta[3])
        depth_imbalance = float(current_meta[4])
        funding_rate = float(current_meta[5])
        order_imbalance = float(current_meta[6])
        realized_vol = float(current_meta[7])

        leverage_cap = self.risk.target_leverage(predicted_vol=realized_vol, realized_vol=realized_vol)

        if size_multiplier != 0.0:
            qty = size_multiplier * self.cfg.base_order_size * leverage_cap
            price_offset = self.cfg.price_offsets[price_offset_idx]
            limit_price = price + price_offset * self.cfg.price_tick
            qty = self.risk.enforce_position_limits(
                position=self._position,
                order_qty=qty,
                mark_price=price,
                leverage_cap=leverage_cap,
            )
            if abs(qty) > 1e-8:
                fill_prob = self._fill_probability(
                    trade_side=np.sign(qty),
                    depth_bid=depth_bid,
                    depth_ask=depth_ask,
                    spread=spread,
                    imbalance=depth_imbalance,
                    order_imbalance=order_imbalance,
                    price_offset=price_offset,
                )
                if self.rng.random() < fill_prob:
                    executed = True
                    self._position += qty
                    fee_rate = (
                        self.cfg.taker_fee_rate if price_offset <= 0 else self.cfg.maker_fee_rate
                    )
                    slippage = limit_price * (self.cfg.slippage_bps / 10_000.0) * self.rng.uniform(-1.0, 1.0)
                    exec_price = limit_price + slippage
                    trade_notional = qty * exec_price
                    fee = abs(trade_notional) * fee_rate
                    self._cash -= fee
                    self._realized_pnl -= fee
                    trade_pnl -= fee
                    self._trades_executed += 1
                    self.reward_tracker.register_trade()

        next_idx = min(self._idx + 1, self._episode_end)
        next_price = float(self.meta[next_idx, 0])
        price_move = next_price - price
        unrealized = self._position * price_move
        funding = self.risk.funding_pnl_adjustment(
            position=self._position,
            mark_price=price,
            funding_rate=funding_rate,
            hours=1.0 / 24.0,
        )
        self._cash += funding
        self._realized_pnl += funding
        trade_pnl += funding

        self._prev_equity = self._equity
        self._equity = self._cash + self._position * next_price
        self.reward_tracker.step(self._equity)
        self.risk.update_drawdown(self._equity)

        reward = float(self._equity - self._prev_equity)
        reward += trade_pnl

        self._idx = next_idx
        terminated = self._idx >= self._episode_end - 1
        truncated = False

        info = {
            "equity": self._equity,
            "position": self._position,
            "leverage_cap": leverage_cap,
            "executed": executed,
            "funding_rate": funding_rate,
            "realized_vol": realized_vol,
        }

        if terminated:
            terminal_bonus, metrics = self.reward_tracker.terminal_reward()
            reward += terminal_bonus
            info.update(metrics)
            info["drawdown"] = self.risk.drawdown

        obs = self._build_observation()
        return obs, reward, terminated, truncated, info

    def _build_observation(self) -> Dict[str, np.ndarray]:
        agent_features = np.array(
            [
                self._position,
                self._cash,
                self._equity,
                self.meta[self._idx, 5],  # funding rate
                self.meta[self._idx, 7],  # realized volatility
                self._trades_executed,
            ],
            dtype=np.float32,
        )
        return {
            "micro": self.micro[self._idx].astype(np.float32),
            "macro": self.macro[self._idx].astype(np.float32),
            "agent": agent_features,
        }

    def _fill_probability(
        self,
        *,
        trade_side: float,
        depth_bid: float,
        depth_ask: float,
        spread: float,
        imbalance: float,
        order_imbalance: float,
        price_offset: int,
    ) -> float:
        depth = depth_bid if trade_side > 0 else depth_ask
        base_prob = np.tanh(abs(order_imbalance)) * 0.5 + 0.25
        spread_penalty = np.exp(-abs(spread) / max(self.cfg.price_tick, 1e-6))
        offset_penalty = np.exp(-abs(price_offset))
        depth_bonus = np.clip(depth / (depth + 100.0), 0.0, 1.0)
        imbalance_bias = 0.5 * (1.0 + trade_side * imbalance)
        prob = base_prob * spread_penalty * offset_penalty * (0.5 + 0.5 * depth_bonus) * imbalance_bias
        return float(np.clip(prob, 0.0, 1.0))
