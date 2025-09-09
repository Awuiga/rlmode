from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass
import os
from typing import Dict, Optional, Tuple, List

from ..common.config import AppConfig
from ..common.logging import get_logger
from ..common.redis_stream import RedisStream
from ..common.schema import Side, TIF, Order, Fill, MarketEvent, Metric
from ..common.utils import utc_ms
from .base import ExchangeAdapter


log = get_logger("paper_exchange")


@dataclass
class SimStats:
    maker_fills: int = 0
    taker_fills: int = 0
    fills_total: int = 0
    slippage_acc_bps: float = 0.0
    markout_acc_bps: float = 0.0


class PaperExchange(ExchangeAdapter):
    def __init__(self, *, cfg: AppConfig.ExecutionSimulatorConfig, rs: RedisStream, symbols_meta: Dict[str, Dict[str, float]]):
        self.cfg = cfg
        self.rs = rs
        self._symbols = symbols_meta
        self._orders: Dict[str, Order] = {}
        self._stats = SimStats()
        self._last_md: Dict[str, MarketEvent] = {}
        self._stop = threading.Event()
        self._bg = threading.Thread(target=self._md_loop, daemon=True)
        self._bg.start()
        random.seed(cfg.seed)

    @property
    def name(self) -> str:
        return "paper"

    # ExchangeAdapter
    def place_limit_post_only(self, *, symbol: str, side: Side, price: float, qty: float, tif: TIF) -> Order:
        cid = f"paper_{utc_ms()}_{random.randint(0, 9999)}"
        order = Order(
            client_id=cid,
            symbol=symbol,
            side=side,
            price=price,
            qty=qty,
            tif=tif,
            post_only=True,
            state="NEW",
        )
        self._orders[cid] = order
        return order

    def cancel(self, *, symbol: str, client_id: str) -> bool:
        return self._orders.pop(client_id, None) is not None

    def get_min_steps(self, *, symbol: str) -> Dict[str, float]:
        meta = self._symbols.get(symbol)
        if not meta:
            return {"tick_size": 0.01, "lot_step": 0.001, "min_qty": 0.001}
        return {
            "tick_size": float(meta.get("tick_size", 0.01)),
            "lot_step": float(meta.get("lot_step", 0.001)),
            "min_qty": float(meta.get("min_qty", 0.001)),
        }

    # Background MD consumer to keep a fresh snapshot for simulation
    def _md_loop(self):
        try:
            group = "paperex"
            consumer = f"c{random.randint(1,9999)}"
            streams = ["md:raw"]
            while not self._stop.is_set():
                msgs = self.rs.read_group(group=group, consumer=consumer, streams=streams, block_ms=1000, count=100)
                for stream, items in msgs:
                    for msg_id, data in items:
                        try:
                            ev = MarketEvent.model_validate(data)
                            self._last_md[ev.symbol] = ev
                        except Exception:
                            pass
                        finally:
                            self.rs.ack(stream, group, msg_id)
        except Exception as e:
            log.error("paperex_md_loop_error", error=str(e))

    # Helpers exposed for unit tests
    def _liq_degrade_bps(self, notional: float) -> float:
        xs = self.cfg.liquidity_degrade_curve.x
        ys = self.cfg.liquidity_degrade_curve.y_bps
        if not xs or not ys:
            return 0.0
        for i, x in enumerate(xs):
            if notional <= x:
                return ys[i]
        return ys[-1]

    def _maker_fill_probability(self, symbol: str, side: Side) -> float:
        ev = self._last_md.get(symbol)
        if not ev:
            return 0.05
        # Compute simple QI
        topk = 5
        bq = sum(q for _, q in ev.bids[:topk])
        aq = sum(q for _, q in ev.asks[:topk])
        denom = (bq + aq) or 1.0
        qi = (bq - aq) / denom
        base = 0.05
        boost = self.cfg.maker_fill_boost_if_qi if ((side == Side.BUY and qi > 0) or (side == Side.SELL and qi < 0)) else 0.0
        # Wider spread lowers maker chance
        spread = max(0.0, ev.ask1 - ev.bid1)
        penalty = min(0.03, spread / ((ev.bid1 + ev.ask1) / 2.0 + 1e-9))
        return max(0.01, min(0.9, base + boost - penalty))

    def _sleep_latency(self):
        base = self.cfg.base_latency_ms
        lo, hi = self.cfg.latency_jitter_ms
        delay = (base + random.uniform(lo, hi)) / 1000.0
        time.sleep(delay)
        return delay

    def simulate_and_publish(self, *, order: Order, is_last_in_ladder: bool = False) -> None:
        # Apply latency
        self._sleep_latency()
        symbol = order.symbol
        side = order.side
        qty_left = order.qty
        notional = order.price * order.qty
        maker_prob = self._maker_fill_probability(symbol, side)
        chunk_min = max(self.cfg.min_partial_fill_qty, 0.01)
        cancel_deadline = time.time() + self.cfg.cancel_timeout_ms / 1000.0
        filled_any = False
        is_maker = True

        while qty_left > 0 and time.time() < cancel_deadline:
            # Per-chunk attempt
            p = maker_prob
            if random.random() < p:
                # Partial maker fill
                fill_qty = max(chunk_min * order.qty, min(qty_left, order.qty))
                qty_left -= fill_qty
                fee = order.price * fill_qty * (self.cfg.fee_bps_maker / 10000.0)
                markout_bps = self._markout_bps(symbol, side)
                self._publish_fill(order, fill_qty, order.price, fee, markout_bps, is_maker=True)
                filled_any = True
                continue
            # No fill this tick; wait a bit
            time.sleep(0.02)

        if qty_left > 0 and (self.cfg.allow_taker_on_timeout or is_last_in_ladder):
            # Taker fallback with slippage
            slip = random.uniform(*self.cfg.taker_slip_bps)
            slip += self._liq_degrade_bps(notional)
            price = order.price * (1 + (slip / 10000.0) * (1 if side == Side.BUY else -1))
            fee = 0.0  # assume taker fee not modeled for paper
            self._publish_fill(order, qty_left, price, fee, self._markout_bps(symbol, side), is_maker=False)
            qty_left = 0.0
            filled_any = True

        # Cancel remaining qty
        if qty_left > 0:
            try:
                self.cancel(symbol=symbol, client_id=order.client_id)
            except Exception:
                pass

    def _markout_bps(self, symbol: str, side: Side) -> float:
        # Occasional negative shock
        if random.random() < self.cfg.shock_prob:
            lo, hi = self.cfg.shock_return
            shock = random.uniform(min(lo, hi), max(lo, hi))
            return shock * 10000.0
        # Otherwise small near-zero
        return random.uniform(-1.0, 1.0)

    def _publish_fill(self, order: Order, qty: float, price: float, fee: float, markout_bps: float, is_maker: bool):
        # In CI/dry-run, allow forcing losses
        force_losses = True if (os.environ.get("DRY_RUN_FORCE_LOSSES") == "1") else False  # type: ignore[name-defined]
        pnl_val = price * qty * (markout_bps / 10000.0) * (1 if order.side == Side.BUY else -1)
        if force_losses:
            pnl_val = -abs(pnl_val) if pnl_val != 0 else -price * qty * 0.0001
        fill = Fill(
            order_id=order.client_id,
            symbol=order.symbol,
            side=order.side,
            price=price,
            qty=qty,
            fee=fee,
            pnl=pnl_val,
            markout=markout_bps,
            is_maker=is_maker,
            ts=utc_ms(),
        )
        self.rs.xadd("exec:fills", fill)
        self.rs.xadd("metrics:executor", Metric(name="fills_total", value=1.0))
        if is_maker:
            self.rs.xadd("metrics:executor", Metric(name="maker_fills_total", value=1.0))
        # Track stats for averages
        self._stats.fills_total += 1
        self._stats.markout_acc_bps += markout_bps
        slip_bps = (abs(price - order.price) / (order.price + 1e-9)) * 10000.0
        self._stats.slippage_acc_bps += slip_bps
        if self._stats.fills_total % 10 == 0:
            self.rs.xadd("metrics:executor", Metric(name="avg_markout_bps", value=self._stats.markout_acc_bps / self._stats.fills_total))
            self.rs.xadd("metrics:executor", Metric(name="avg_slippage_bps", value=self._stats.slippage_acc_bps / self._stats.fills_total))
