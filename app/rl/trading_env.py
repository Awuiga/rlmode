"""Gymnasium environment simulating crypto trading with realistic frictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


@dataclass
class TradingEnvConfig:
    """Configuration parameters for the trading environment."""

    initial_capital: float = 100_000.0
    maker_fee_rate: float = 0.0005  # 0.05%
    slippage_pct: float = 0.001  # +/-0.1%
    max_trades_per_day: int = 100
    max_position: float = 5.0  # in coins
    max_steps: int = 24 * 60  # minute-level day length
    price_drift: float = 0.0002
    price_vol: float = 0.005


class TradingEnv(gym.Env):
    """Simple crypto trading simulator with commissions, slippage and trade caps.

    Action space:
        0 -> buy (open/increase long)
        1 -> sell (reduce/short)
        2 -> hold (no trade)

    Observation space (basic example):
        [price_return, position, cash_ratio, trades_used / max_trades]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[TradingEnvConfig] = None) -> None:
        super().__init__()
        self.cfg = config or TradingEnvConfig()

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng()
        self.reset()

    # ------------------------------------------------------------------ #
    # Core environment API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step: int = 0
        self._price: float = 25_000.0
        self._prev_price: float = self._price
        self._position: float = 0.0
        self._cash: float = self.cfg.initial_capital
        self._equity: float = self.cfg.initial_capital
        self._prev_equity: float = self._equity
        self._trade_count: int = 0
        self._last_reward: float = 0.0

        observation = self._build_observation()
        info = {"equity": self._equity, "price": self._price}
        return observation, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        self._step += 1
        self._prev_equity = self._equity
        self._prev_price = self._price

        # Simulate price move with drift + stochastic volatility
        price_return = (
            self.cfg.price_drift + self.cfg.price_vol * self._rng.normal()
        )
        self._price = max(1.0, self._price * (1.0 + price_return))

        executed = False
        commission = 0.0
        slippage = 0.0
        position_delta = 0.0

        if action != 2 and self._trade_count < self.cfg.max_trades_per_day:
            executed = True
            self._trade_count += 1

            qty = 1.0 if action == 0 else -1.0
            next_position = self._position + qty
            next_position = float(
                np.clip(next_position, -self.cfg.max_position, self.cfg.max_position)
            )
            position_delta = next_position - self._position

            # Apply slippage by adjusting execution price.
            slip = self.cfg.slippage_pct * self._rng.uniform(-1.0, 1.0)
            exec_price = self._price * (1.0 + slip)
            slippage = abs(exec_price - self._price)

            trade_notional = position_delta * exec_price
            commission = abs(trade_notional) * self.cfg.maker_fee_rate

            self._cash -= trade_notional + commission
            self._position = next_position

        # MTM valuation of position
        self._equity = self._cash + self._position * self._price

        reward = (self._equity - self._prev_equity) - commission
        self._last_reward = reward

        terminated = self._step >= self.cfg.max_steps
        truncated = False

        observation = self._build_observation()
        info = {
            "equity": self._equity,
            "position": self._position,
            "price": self._price,
            "commission": commission,
            "slippage": slippage,
            "executed": executed,
            "trades": self._trade_count,
        }
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Rendering & helpers
    # ------------------------------------------------------------------ #

    def render(self):
        print(
            f"[step={self._step:4d}] price={self._price:8.2f} "
            f"position={self._position:+.2f} equity={self._equity:,.2f} "
            f"reward={self._last_reward:+.2f}"
        )

    def close(self):
        pass

    def _build_observation(self) -> np.ndarray:
        price_ret = (self._price - self._prev_price) / self._prev_price if self._prev_price else 0.0
        cash_ratio = self._cash / (self._equity + 1e-8)
        trades_ratio = self._trade_count / max(1, self.cfg.max_trades_per_day)
        obs = np.array(
            [
                price_ret,
                self._position / max(1.0, self.cfg.max_position),
                cash_ratio,
                trades_ratio,
            ],
            dtype=np.float32,
        )
        return obs
