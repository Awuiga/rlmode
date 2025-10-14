from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


@dataclass
class RewardConfig:
    """Configuration for daily reward shaping."""

    hindsight_weight: float = 1.0
    trade_penalty: float = 0.01
    drawdown_penalty: float = 0.0
    positive_pnl_threshold: float = 0.0


@dataclass
class EpisodeStats:
    """Rolling statistics over one trading day."""

    equity_start: float = 0.0
    equity_end: float = 0.0
    max_equity: float = 0.0
    min_equity: float = 0.0
    trade_count: int = 0
    hindsight_bonus: float = 0.0
    info: Dict[str, float] = field(default_factory=dict)


class DailyRewardCalculator:
    """Reward helper that focuses on end-of-day performance."""

    def __init__(self, cfg: RewardConfig) -> None:
        self.cfg = cfg
        self._stats = EpisodeStats()
        self._last_equity: Optional[float] = None

    def reset(self, equity: float) -> None:
        self._stats = EpisodeStats(
            equity_start=equity,
            equity_end=equity,
            max_equity=equity,
            min_equity=equity,
        )
        self._last_equity = equity

    def register_trade(self) -> None:
        self._stats.trade_count += 1

    def step(self, equity: float) -> float:
        if self._last_equity is None:
            self._last_equity = equity
        pnl_delta = float(equity - self._last_equity)
        self._last_equity = equity
        self._stats.equity_end = equity
        self._stats.max_equity = max(self._stats.max_equity, equity)
        self._stats.min_equity = min(self._stats.min_equity, equity)
        return pnl_delta

    def finalize(self) -> float:
        reward = -self.cfg.trade_penalty * self._stats.trade_count
        if self.cfg.drawdown_penalty > 0.0:
            drawdown = max(0.0, self._stats.max_equity - self._stats.min_equity)
            reward -= self.cfg.drawdown_penalty * drawdown

        daily_pnl = self._stats.equity_end - self._stats.equity_start
        if daily_pnl > self.cfg.positive_pnl_threshold:
            bonus = self.cfg.hindsight_weight * (daily_pnl - self.cfg.positive_pnl_threshold)
            self._stats.hindsight_bonus = bonus
            reward += bonus
        else:
            self._stats.hindsight_bonus = 0.0

        self._stats.info = {
            "daily_pnl": daily_pnl,
            "trade_count": float(self._stats.trade_count),
            "drawdown": float(self._stats.max_equity - self._stats.min_equity),
            "hindsight_bonus": self._stats.hindsight_bonus,
        }
        return reward

    @property
    def stats(self) -> EpisodeStats:
        return self._stats


class DailyPnLEnv(gym.Env):
    """Simple Gym environment that uses the daily reward scheme."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        observation_shape: Tuple[int, ...] = (128,),
        action_space_size: int = 3,
        reward_config: RewardConfig | None = None,
        max_episode_steps: int = 1440,
    ) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation_shape,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(action_space_size)
        self._reward = DailyRewardCalculator(reward_config or RewardConfig())
        self.max_episode_steps = max_episode_steps
        self._rng = np.random.default_rng()

        self._equity = 0.0
        self._step_count = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._equity = self._initial_equity()
        self._step_count = 0
        self._reward.reset(self._equity)
        return self._next_observation(), {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        self._step_count += 1
        filled = self._execute_action(action)
        if filled and action != 0:
            self._reward.register_trade()

        self._equity = self._mark_to_market(action)
        reward = self._reward.step(self._equity)

        terminated = self._is_end_of_day()
        truncated = self._step_count >= self.max_episode_steps
        if terminated or truncated:
            reward += self._reward.finalize()

        obs = self._next_observation()
        info = {"equity": self._equity, **self._reward.stats.info}
        return obs, reward, terminated, truncated, info

    # --- domain specific hooks ---------------------------------------------

    def _initial_equity(self) -> float:
        return 1_000_000.0

    def _mark_to_market(self, action: int) -> float:
        noise = self._rng.normal(0.0, 5.0)
        return float(self._equity + noise)

    def _execute_action(self, action: int) -> bool:
        return action != 0

    def _next_observation(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _is_end_of_day(self) -> bool:
        return False
