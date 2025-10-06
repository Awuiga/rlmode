from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Tuple

from ..common.schema import Side


@dataclass
class SymbolRef:
    lot_step: float
    min_qty: float
    min_notional: float
    tick_size: float
    mmr: float


def _round_to_step(value: float, step: float, mode: str = "floor") -> float:
    if step <= 0:
        return value
    scaled = value / step
    if mode == "ceil":
        scaled = math.ceil(scaled - 1e-12)
    elif mode == "round":
        scaled = round(scaled)
    else:  # floor (default)
        scaled = math.floor(scaled + 1e-12)
    return max(0.0, scaled * step)


def compute_order_qty(
    price: float,
    free_balance_usd: float,
    ref: SymbolRef,
    leverage: float,
    notional_cap_usd: float,
    *,
    lot_size_rounding: str = "floor",
    min_free_balance_usd: float = 0.0,
) -> float:
    if price <= 0 or free_balance_usd <= 0 or leverage <= 0 or notional_cap_usd <= 0:
        return 0.0
    if free_balance_usd < min_free_balance_usd:
        return 0.0
    effective_notional_cap = min(notional_cap_usd, free_balance_usd * leverage)
    if effective_notional_cap <= 0:
        return 0.0
    qty_raw = effective_notional_cap / price
    qty = _round_to_step(qty_raw, ref.lot_step or 1.0, lot_size_rounding)
    if qty < ref.min_qty or qty * price < ref.min_notional:
        return 0.0
    return qty


def est_liq_price_long(entry_price: float, leverage: float, mmr: float) -> float:
    if entry_price <= 0 or leverage <= 0:
        return 0.0
    # Approximate long liquidation price (isolated margin): entry * (1 - 1/leverage + mmr)
    return max(0.0, entry_price * (1.0 - (1.0 / leverage) + mmr))


def near_liq_block(
    entry_price: float,
    leverage: float,
    mmr: float,
    assumed_sl_pct: float,
    sl_buffer_multiplier: float,
) -> bool:
    if entry_price <= 0 or leverage <= 0:
        return True
    if assumed_sl_pct <= 0:
        return False
    liq_price = est_liq_price_long(entry_price, leverage, mmr)
    if liq_price <= 0 or liq_price >= entry_price:
        # no margin left / already liquidated -> block
        return True
    distance_pct = (entry_price - liq_price) / entry_price
    threshold = assumed_sl_pct * max(sl_buffer_multiplier, 0.0)
    return distance_pct <= threshold


def compute_side_depth(features: Mapping[str, float], side: Side, levels: int) -> float:
    base_key = "depth_bids" if side == Side.BUY else "depth_asks"
    depth_values = []
    for lvl in range(1, max(levels, 0) + 1):
        key = f"{base_key}_l{lvl}"
        if key in features:
            depth_values.append(float(features.get(key, 0.0)))
    if not depth_values:
        depth_values.append(float(features.get(base_key, 0.0)))
    return max(0.0, sum(depth_values))


def evaluate_liquidity_by_side(
    qty: float,
    side: Side,
    features: Mapping[str, float],
    alpha: float,
    levels: int,
) -> Tuple[bool, float, float]:
    required = max(0.0, alpha * qty)
    available = compute_side_depth(features, side, levels)
    return available >= required, required, available
