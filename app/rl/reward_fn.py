from __future__ import annotations

from typing import Final


HINDSIGHT_BONUS_DEFAULT: Final[float] = 10.0
TRADE_PENALTY_RATE: Final[float] = 0.01
DRAWDOWN_PENALTY_RATE: Final[float] = 0.1


def calculate_reward(
    *,
    profit: float,
    previous_profit: float,
    trades_per_day: int,
    max_drawdown: float,
    done: bool,
    hindsight_bonus: float = HINDSIGHT_BONUS_DEFAULT,
    trade_penalty_rate: float = TRADE_PENALTY_RATE,
    drawdown_penalty_rate: float = DRAWDOWN_PENALTY_RATE,
) -> float:
    """Compute step reward for a trading agent compatible with Gym.

    Args:
        profit: Current equity or cumulative profit for the episode.
        previous_profit: Equity/profit observed on the previous step.
        trades_per_day: Number of executed trades in the current episode.
        max_drawdown: Maximum absolute drawdown reached during the episode.
        done: Whether the current step is terminal (end of episode/day).
        hindsight_bonus: Bonus applied at the end of the episode if profit > 0.
        trade_penalty_rate: Penalty multiplier for executed trades.
        drawdown_penalty_rate: Penalty multiplier for drawdown size.

    Returns:
        Scalar reward to be returned by the Gym environment step().
    """
    # Base reward is the change in capital since the previous step.
    reward = float(profit - previous_profit)

    if done:
        # Reward agent for finishing the day with positive PnL.
        if profit > 0.0:
            reward += float(hindsight_bonus)

        # Penalise excessive trading activity.
        reward -= trade_penalty_rate * float(trades_per_day)

        # Penalise large drawdowns to encourage smoother equity curves.
        reward -= drawdown_penalty_rate * float(max_drawdown)

    return reward
