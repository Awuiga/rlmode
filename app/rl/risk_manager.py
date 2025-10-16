"""Risk management utilities for leveraged RL trading agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True)
class RiskParams:
    base_leverage: float = 2.0
    max_leverage: float = 5.0
    min_leverage: float = 1.0
    liquidation_buffer: float = 0.15
    max_position_notional: float = 150_000.0
    loss_cap: float = 0.05  # relative to equity
    funding_sensitivity: float = 0.5
    lookback_vol_target: float = 0.015
    leverage_vol_slope: float = -25.0


class RiskManager:
    """Derive leverage caps and enforce liquidation buffers."""

    def __init__(self, params: RiskParams) -> None:
        self.params = params
        self._prev_high_watermark: Optional[float] = None
        self._realized_drawdown: float = 0.0

    def reset(self, equity: float) -> None:
        self._prev_high_watermark = equity
        self._realized_drawdown = 0.0

    def target_leverage(self, predicted_vol: float, realized_vol: float) -> float:
        vol = max(predicted_vol, 1e-6)
        deviation = realized_vol - self.params.lookback_vol_target
        leverage_adj = self.params.leverage_vol_slope * deviation
        leverage = self.params.base_leverage + leverage_adj
        leverage = np.clip(leverage, self.params.min_leverage, self.params.max_leverage)
        if vol > 0:
            leverage = min(leverage, self.params.lookback_vol_target / vol * self.params.base_leverage)
        return float(leverage)

    def check_liquidation_buffer(
        self,
        *,
        equity: float,
        position: float,
        mark_price: float,
        liquidation_price: float,
    ) -> bool:
        if position == 0.0:
            return True
        buffer = abs((mark_price - liquidation_price) / mark_price)
        return buffer >= self.params.liquidation_buffer

    def enforce_position_limits(
        self,
        *,
        position: float,
        order_qty: float,
        mark_price: float,
        leverage_cap: float,
    ) -> float:
        desired_position = position + order_qty
        notional = abs(desired_position) * mark_price
        max_notional = min(self.params.max_position_notional, leverage_cap * mark_price)
        if notional <= max_notional:
            return order_qty
        allowed = np.clip(max_notional / mark_price - abs(position), 0.0, abs(order_qty))
        return np.sign(order_qty) * allowed

    def update_drawdown(self, equity: float) -> None:
        if self._prev_high_watermark is None:
            self._prev_high_watermark = equity
        if equity > self._prev_high_watermark:
            self._prev_high_watermark = equity
        drawdown = (equity - (self._prev_high_watermark or equity)) / (self._prev_high_watermark or 1.0)
        self._realized_drawdown = min(self._realized_drawdown, drawdown)

    @property
    def drawdown(self) -> float:
        return self._realized_drawdown

    def funding_pnl_adjustment(self, position: float, mark_price: float, funding_rate: float, hours: float) -> float:
        accrual = position * mark_price * funding_rate * hours * self.params.funding_sensitivity
        return -float(accrual)

