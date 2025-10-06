from __future__ import annotations

from typing import List, Tuple

from ..common.schema import Side
from ..common.utils import round_to_step, clamp_min_qty


def build_ladder(
    *,
    side: Side,
    mid_price: float,
    qty: float,
    fractions: List[float],
    step_ticks: int,
    tick_size: float,
    lot_step: float,
    min_qty: float,
    use_percent: bool = False,
) -> List[Tuple[float, float]]:
    orders: List[Tuple[float, float]] = []
    # Base price: inside spread for maker bias
    for i, f in enumerate(fractions):
        raw_qty = qty * f
        q = round_to_step(raw_qty, lot_step)
        q = clamp_min_qty(q, min_qty)
        if q <= 0:
            continue
        if use_percent:
            step_price = mid_price * (1 + (step_ticks * (i + 0.5)) / 100.0)
        else:
            step_price = mid_price
        if side == Side.BUY:
            price = step_price - (i + 1) * step_ticks * tick_size
        else:
            price = step_price + (i + 1) * step_ticks * tick_size
        price = round_to_step(price, tick_size)
        orders.append((price, q))
    return orders

