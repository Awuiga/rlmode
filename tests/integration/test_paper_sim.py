import os

import pytest

pytestmark = pytest.mark.integration

from app.exchange.paper import PaperExchange
from app.common.redis_stream import RedisStream
from app.common.schema import Side, TIF, Order, MarketEvent, LastTrade
from app.common.config import AppConfig


def make_cfg():
    return AppConfig.ExecutionSimulatorConfig(
        seed=1,
        base_latency_ms=1,
        latency_jitter_ms=[0, 0],
        maker_fill_model="queue",
        taker_slip_bps=[1.0, 4.0],
        maker_fill_boost_if_qi=0.3,
        min_partial_fill_qty=0.25,
        cancel_timeout_ms=10,
        shock_prob=0.0,
        shock_return=[-0.1, -0.1],
    )


def test_liquidity_degrade_curve_slippage_increases():
    rs = RedisStream("fakeredis://")
    cfg = make_cfg()
    ex = PaperExchange(cfg=cfg, rs=rs, symbols_meta={"BTCUSDT": {"tick_size": 0.1, "lot_step": 0.001, "min_qty": 0.001}})
    low = ex._liq_degrade_bps(1000)
    mid = ex._liq_degrade_bps(100000)
    high = ex._liq_degrade_bps(1500000)
    assert low <= mid <= high


def test_qi_boosts_maker_prob():
    rs = RedisStream("fakeredis://")
    cfg = make_cfg()
    ex = PaperExchange(cfg=cfg, rs=rs, symbols_meta={})
    # Feed two md snapshots with different QI
    md_pos = MarketEvent(ts=1, symbol="BTCUSDT", bid1=100.0, ask1=100.1, bids=[(99.9, 10.0)], asks=[(100.1, 1.0)], last_trade=LastTrade(price=100.0, qty=1.0, side="buy"))
    md_neg = MarketEvent(ts=2, symbol="BTCUSDT", bid1=100.0, ask1=100.1, bids=[(99.9, 1.0)], asks=[(100.1, 10.0)], last_trade=LastTrade(price=100.0, qty=1.0, side="sell"))
    ex._last_md["BTCUSDT"] = md_pos
    p_buy = ex._maker_fill_probability("BTCUSDT", Side.BUY)
    ex._last_md["BTCUSDT"] = md_neg
    p_buy_neg = ex._maker_fill_probability("BTCUSDT", Side.BUY)
    assert p_buy > p_buy_neg


def test_shock_negative_markout():
    rs = RedisStream("fakeredis://")
    cfg = make_cfg()
    cfg.shock_prob = 1.0
    cfg.shock_return = [-0.10, -0.10]
    ex = PaperExchange(cfg=cfg, rs=rs, symbols_meta={})
    mo = ex._markout_bps("BTCUSDT", Side.BUY)
    assert mo < 0

