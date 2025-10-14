"""Performance metrics utilities for trading agents."""

from __future__ import annotations

from math import sqrt
from typing import Dict, Iterable, Optional, Sequence

import numpy as np


def calculate_metrics(
    equity_curve: Sequence[float],
    *,
    episode: int = 1,
    print_every: int = 1,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Compute key performance metrics for an equity curve.

    Args:
        equity_curve: Sequence of cumulative equity values for the episode.
        episode: Index of the current training episode (1-based).
        print_every: Print metrics when ``episode % print_every == 0``.
        risk_free_rate: Optional risk-free rate per period for Sharpe/Sortino.

    Returns:
        Dictionary with sharpe, sortino, max_drawdown, winrate.
    """
    equity = np.asarray(equity_curve, dtype=np.float64)
    if equity.ndim != 1 or equity.size < 2:
        raise ValueError("equity_curve must be 1-D with at least two points")

    returns = np.diff(equity) / np.clip(equity[:-1], 1e-8, None)
    excess_returns = returns - risk_free_rate

    sharpe = _safe_ratio(excess_returns.mean(), excess_returns.std(ddof=1))
    downside = excess_returns[excess_returns < 0.0]
    downside_std = downside.std(ddof=1) if downside.size > 0 else 0.0
    sortino = _safe_ratio(excess_returns.mean(), downside_std)

    rolling_max = np.maximum.accumulate(equity)
    drawdowns = (equity - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min()) if drawdowns.size else 0.0

    wins = np.count_nonzero(returns > 0.0)
    total_trades = returns.size
    winrate = float(wins / total_trades) if total_trades else 0.0

    metrics = {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "winrate": float(winrate),
    }

    if print_every > 0 and episode % print_every == 0:
        print(
            f"[episode={episode}] "
            f"Sharpe={metrics['sharpe']:.4f} "
            f"Sortino={metrics['sortino']:.4f} "
            f"MDD={metrics['max_drawdown']:.4%} "
            f"Winrate={metrics['winrate']:.2%}"
        )

    return metrics


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)

