from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.common.schema import Side
from app.executor.sizing import (
    SymbolRef,
    compute_order_qty,
    evaluate_liquidity_by_side,
)


def test_liquidity_by_side_gate():
    features = {"depth_bids": 0.2}
    passed, required, available = evaluate_liquidity_by_side(
        qty=0.5,
        side=Side.BUY,
        features=features,
        alpha=1.5,
        levels=3,
    )
    assert not passed
    assert available < required


def test_qty_zero_reason():
    ref = SymbolRef(lot_step=0.1, min_qty=0.5, min_notional=50.0, tick_size=0.05, mmr=0.004)
    qty = compute_order_qty(
        price=100.0,
        free_balance_usd=20.0,
        ref=ref,
        leverage=5.0,
        notional_cap_usd=200.0,
        lot_size_rounding="floor",
        min_free_balance_usd=0.0,
    )
    assert qty == 0.0
    drop_reason = "qty_zero" if qty == 0 else None
    assert drop_reason == "qty_zero"
