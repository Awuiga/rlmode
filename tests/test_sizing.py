from __future__ import annotations

import math

import pytest

from app.executor.sizing import (
    SymbolRef,
    compute_order_qty,
    est_liq_price_long,
    near_liq_block,
)


def test_compute_order_qty_flooring():
    ref = SymbolRef(lot_step=0.001, min_qty=0.001, min_notional=10.0, tick_size=0.1, mmr=0.004)
    qty = compute_order_qty(
        price=30000.0,
        free_balance_usd=500.0,
        ref=ref,
        leverage=10.0,
        notional_cap_usd=20000.0,
        lot_size_rounding="floor",
        min_free_balance_usd=100.0,
    )
    assert math.isclose(qty, 0.166, rel_tol=1e-9)


def test_min_notional_and_min_qty():
    ref = SymbolRef(lot_step=0.1, min_qty=0.5, min_notional=50.0, tick_size=0.05, mmr=0.004)
    qty = compute_order_qty(
        price=100.0,
        free_balance_usd=30.0,
        ref=ref,
        leverage=5.0,
        notional_cap_usd=1000.0,
        lot_size_rounding="floor",
        min_free_balance_usd=0.0,
    )
    assert qty == 0.0


def test_near_liq_block():
    entry = 20000.0
    leverage = 10.0
    mmr = 0.005
    liq_price = est_liq_price_long(entry, leverage, mmr)
    assert liq_price < entry
    # distance roughly 9.5%, larger than threshold -> not blocked
    assert near_liq_block(entry, leverage, mmr, assumed_sl_pct=0.02, sl_buffer_multiplier=1.5) is False
    # tighten stop -> should block
    assert near_liq_block(entry, leverage, mmr, assumed_sl_pct=0.01, sl_buffer_multiplier=1.0) is True
