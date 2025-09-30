from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from typing import Dict, Mapping, Optional, Set, Tuple

import yaml

from ..common.config import load_app_config
from ..common.logging import setup_logging, get_logger
from ..common.redis_stream import RedisStream
from ..common.schema import ApprovedSignal, Order, Fill, TIF, Side, Metric
from ..common.utils import utc_ms
from .ladder import build_ladder
from .positions import Position
from .sizing import (
    SymbolRef,
    compute_order_qty,
    evaluate_liquidity_by_side,
    near_liq_block,
)
from ..exchange import create_exchange, PaperExchange
from ..exchange.base import ExchangeAdapter


log = get_logger("executor")


def load_symbols_meta(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def probability_bucket(prob: float) -> str:
    if prob < 0.25:
        return "lt_0.25"
    if prob < 0.5:
        return "0.25_0.5"
    if prob < 0.75:
        return "0.5_0.75"
    return "gte_0.75"


def estimate_fill_probability(feats: Dict[str, float], side: Side, level_idx: int, cfg) -> float:
    queue_proxy = float(feats.get("queue_position_proxy", feats.get("qi", 0.5)))
    cap = getattr(cfg, "queue_proxy_cap", 1.0) or 1.0
    queue_norm = max(0.0, min(1.0, queue_proxy / cap))
    queue_factor = 1.0 - queue_norm
    qi_val = float(feats.get("qi", 0.0))
    if side == Side.BUY:
        qi_factor = max(0.0, min(1.0, (qi_val + 1.0) / 2.0))
    else:
        qi_factor = max(0.0, min(1.0, (1.0 - qi_val) / 2.0))
    depth_key = "depth_bids" if side == Side.BUY else "depth_asks"
    opp_depth_key = "depth_asks" if side == Side.BUY else "depth_bids"
    depth = max(0.0, float(feats.get(depth_key, 0.0)))
    opp_depth = max(0.0, float(feats.get(opp_depth_key, 0.0)))
    depth_ratio = depth / (depth + opp_depth + 1e-9) if depth + opp_depth > 0 else 0.0
    base = getattr(cfg, "base", 0.3)
    queue_weight = getattr(cfg, "queue_weight", 0.35)
    qi_weight = getattr(cfg, "qi_weight", 0.25)
    depth_weight = getattr(cfg, "depth_weight", 0.2)
    ladder_decay = getattr(cfg, "ladder_decay", 0.35)
    spread_penalty = getattr(cfg, "spread_penalty", 0.05)
    spread = float(feats.get("spread", 0.0))
    prob = base
    prob += queue_weight * queue_factor
    prob += qi_weight * qi_factor
    prob += depth_weight * depth_ratio
    prob *= math.exp(-ladder_decay * level_idx)
    prob -= spread_penalty * spread
    min_value = getattr(cfg, "min_value", 0.05)
    max_value = getattr(cfg, "max_value", 0.95)
    return max(min_value, min(max_value, prob))


def _canary_vote(signal_id: str, rng_seed: int) -> float:
    payload = f"{rng_seed}:{signal_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def decide_route_label(
    *,
    signal_id: str,
    rollout_mode: str,
    exchange_mode: str,
    canary_fraction: float,
    rng_seed: int,
    has_real: bool,
    has_paper: bool,
    default_route: str,
) -> str:
    """Select the routing label given rollout flags and available exchanges."""
    rollout = (rollout_mode or "disabled").lower()
    exchange = (exchange_mode or "paper").lower()
    fraction = max(0.0, min(1.0, canary_fraction))
    if rollout == "full" and exchange == "real" and has_real:
        return "full"
    if rollout == "canary" and exchange == "real" and has_real:
        if not has_paper:
            return "real"
        vote = _canary_vote(signal_id, rng_seed)
        return "canary" if vote < fraction else "paper"
    if exchange == "real" and has_real:
        return "real"
    if has_paper:
        return "paper"
    return default_route


def main():
    setup_logging()
    cfg = load_app_config()
    rs = RedisStream(cfg.redis.url, default_maxlen=cfg.redis.streams_maxlen)
    symbols_meta = load_symbols_meta("config/symbols.yml")

    # Choose exchange client based on execution mode
    primary_exchange = create_exchange(cfg, rs, symbols_meta)
    rollout_mode = (cfg.rollout.mode or "disabled").lower()
    exchange_mode = (cfg.exchange.mode or "paper").lower()
    real_exchange: Optional[ExchangeAdapter] = None
    paper_exchange: Optional[PaperExchange] = None
    if isinstance(primary_exchange, PaperExchange):
        paper_exchange = primary_exchange
    else:
        real_exchange = primary_exchange
    if rollout_mode == "canary" and exchange_mode == "real" and real_exchange is not None and paper_exchange is None:
        paper_exchange = PaperExchange(cfg=cfg.execution.simulator, rs=rs, symbols_meta=symbols_meta)
    default_route_label = "paper" if isinstance(primary_exchange, PaperExchange) else "real"

    pos = {sym: Position() for sym in cfg.symbols}
    halted = False
    open_orders: Set[str] = set()
    open_orders_by_symbol: Dict[str, Set[str]] = defaultdict(set)
    order_symbol: Dict[str, str] = {}
    order_exchange: Dict[str, ExchangeAdapter] = {}
    order_route: Dict[str, str] = {}
    last_execution_side: Dict[str, Side] = {}
    last_execution_ts: Dict[str, int] = {}

    symbol_refs: Dict[str, SymbolRef] = {}
    for sym, meta in symbols_meta.items():
        try:
            symbol_refs[sym] = SymbolRef(
                lot_step=float(meta.get("lot_step", 0.001)),
                min_qty=float(meta.get("min_qty", 0.0)),
                min_notional=float(meta.get("min_notional", 0.0)),
                tick_size=float(meta.get("tick_size", 0.0)),
                mmr=float(meta.get("mmr", 0.004)),
            )
        except Exception:
            symbol_refs[sym] = SymbolRef(0.001, 0.0, 0.0, 0.0, 0.004)

    sizing_cfg = cfg.executor.sizing
    liquidity_gate_cfg = cfg.executor.gates.liquidity_by_side
    near_liq_cfg = cfg.risk.near_liq_guard
    assumed_sl_pct_default = float(getattr(cfg.risk, "assumed_sl_pct", 0.0) or 0.0)
    effective_free_balance = max(
        sizing_cfg.min_free_balance_usd,
        sizing_cfg.notional_cap_usd / max(sizing_cfg.leverage, 1e-9),
    )

    log.info("executor_start", exchange=primary_exchange.name)

    group = "exec"
    consumer = "c1"
    streams = ["sig:approved", "control:events"]

    import os
    max_iters_env = int(os.environ.get("DRY_RUN_MAX_ITER", "0")) if "DRY_RUN_MAX_ITER" in os.environ else None
    processed = 0
    while True:
        msgs = rs.read_group(group=group, consumer=consumer, streams=streams, block_ms=1000, count=50)
        if not msgs:
            continue
        for stream, items in msgs:
            for msg_id, data in items:
                try:
                    if stream == "control:events":
                        # Stop new orders; cancel anything open
                        halted = True
                        for oid in list(open_orders):
                            sym_for_order = order_symbol.pop(oid, None)
                            exchange_for_order = order_exchange.pop(oid, primary_exchange)
                            order_route.pop(oid, None)
                            try:
                                exchange_for_order.cancel(symbol=sym_for_order or cfg.symbols[0], client_id=oid)
                            except Exception:
                                pass
                            finally:
                                open_orders.discard(oid)
                                if sym_for_order:
                                    open_orders_by_symbol[sym_for_order].discard(oid)
                        rs.xadd("metrics:executor", Metric(name="open_orders", value=float(len(open_orders))))
                        log.warning("executor_halted")
                    else:
                        if halted:
                            # Do not place new orders, just ack and continue
                            continue
                        sig = ApprovedSignal.model_validate(data)
                        sym = sig.symbol
                        now_ms = utc_ms()
                        prev_side = last_execution_side.get(sym)
                        if prev_side is not None and prev_side != sig.side:
                            required = max(cfg.executor.cooldown_ms, cfg.executor.cancel_latency_p95_ms)
                            elapsed = now_ms - last_execution_ts.get(sym, 0)
                            if elapsed < required:
                                wait_ms = required - elapsed
                                rs.xadd("metrics:executor", Metric(name="cooldown_skip_total", value=1.0, labels={"symbol": sym}))
                                log.info("executor_flip_cooldown", symbol=sym, wait_ms=int(wait_ms), prev=str(prev_side), new=str(sig.side))
                                continue
                        meta = symbols_meta.get(sym, {})
                        scene_id = str(sig.features.get("scene_id") or f"{sym}-scene")
                        tick = float(meta.get("tick_size", 0.01))
                        lot = float(meta.get("lot_step", 0.001))
                        min_qty = float(meta.get("min_qty", 0.001))
                        side_sign = 1 if sig.side == Side.BUY else -1
                        caps = cfg.executor.exposure_caps
                        symbol_orders = open_orders_by_symbol[sym]
                        active_symbols = sum(1 for orders in open_orders_by_symbol.values() if orders)
                        if caps.max_positions and active_symbols >= caps.max_positions and not symbol_orders:
                            rs.xadd("metrics:executor", Metric(name="exposure_block_total", value=1.0, labels={"symbol": sym, "reason": "max_positions"}))
                            log.info("executor_skip_exposure", symbol=sym, reason="max_positions")
                            continue
                        symbol_cap = caps.per_symbol.get(sym)
                        if symbol_cap and len(symbol_orders) >= symbol_cap:
                            rs.xadd("metrics:executor", Metric(name="exposure_block_total", value=1.0, labels={"symbol": sym, "reason": "per_symbol"}))
                            log.info("executor_skip_exposure", symbol=sym, reason="per_symbol_cap")
                            continue
                        qi_value = sig.features.get("qi", 0.0)
                        rs.xadd("metrics:executor", Metric(name="queue_position_proxy", value=qi_value, labels={"symbol": sym}))
                        aggressive = qi_value >= cfg.executor.qi_aggressive_threshold
                        spread_regime = int(sig.features.get("spread_regime", 1))
                        features = sig.features or {}
                        if cfg.executor.microprice_guard:
                            mp_vel = features.get("microprice_velocity", 0.0)
                            if mp_vel * side_sign <= 0:
                                rs.xadd("metrics:executor", Metric(name="skip_microprice_guard", value=1.0, labels={"symbol": sym}))
                                log.info("executor_skip_microprice", symbol=sym, microprice_velocity=mp_vel, side=str(sig.side))
                                continue

                        # Determine anchor price from top of book
                        bid1 = features.get("bid1", 0.0)
                        ask1 = features.get("ask1", 0.0)
                        anchor = bid1 if sig.side == Side.BUY else ask1
                        if anchor <= 0:
                            anchor = features.get("mid", 0.0) or 30000.0

                        # Apply entry_ref offsets (ticks or percent)
                        entry_price = anchor
                        if sig.entry_ref.offset_ticks is not None:
                            if sig.side == Side.BUY:
                                entry_price = anchor - sig.entry_ref.offset_ticks * tick
                            else:
                                entry_price = anchor + sig.entry_ref.offset_ticks * tick
                        elif sig.entry_ref.offset_pct is not None:
                            if sig.side == Side.BUY:
                                entry_price = anchor * (1 - sig.entry_ref.offset_pct)
                            else:
                                entry_price = anchor * (1 + sig.entry_ref.offset_pct)
                        elif sig.entry_ref.price is not None:
                            entry_price = sig.entry_ref.price

                        symbol_ref = symbol_refs.get(sym) or SymbolRef(0.001, 0.0, 0.0, 0.0, 0.004)
                        qty = compute_order_qty(
                            price=entry_price,
                            free_balance_usd=effective_free_balance,
                            ref=symbol_ref,
                            leverage=sizing_cfg.leverage,
                            notional_cap_usd=sizing_cfg.notional_cap_usd,
                            lot_size_rounding=sizing_cfg.lot_size_rounding,
                            min_free_balance_usd=sizing_cfg.min_free_balance_usd,
                        )
                        if qty <= 0:
                            rs.xadd(
                                "metrics:executor",
                                Metric(name="gate_drop_reason_total", value=1.0, labels={"symbol": sym, "reason": "qty_zero"}),
                            )
                            log.info("executor_skip_qty", symbol=sym, reason="qty_zero")
                            continue

                        if near_liq_cfg.enabled and sig.side == Side.BUY:
                            sl_pct = float(getattr(sig, "sl_pct", 0.0) or assumed_sl_pct_default)
                            if sl_pct > 0 and near_liq_block(
                                entry_price,
                                sizing_cfg.leverage,
                                symbol_ref.mmr,
                                sl_pct,
                                near_liq_cfg.sl_buffer_multiplier,
                            ):
                                rs.xadd(
                                    "metrics:executor",
                                    Metric(name="gate_drop_reason_total", value=1.0, labels={"symbol": sym, "reason": "near_liquidation"}),
                                )
                                log.info("executor_skip_near_liq", symbol=sym, entry=entry_price, sl_pct=sl_pct)
                                continue

                        if liquidity_gate_cfg.enabled:
                            passed, required_liq, available_liq = evaluate_liquidity_by_side(
                                qty,
                                sig.side,
                                features,
                                liquidity_gate_cfg.alpha,
                                liquidity_gate_cfg.levels,
                            )
                            rs.xadd(
                                "metrics:executor",
                                Metric(
                                    name="exec_needed_side_liquidity",
                                    value=float(required_liq),
                                    labels={"symbol": sym, "side": sig.side.value},
                                ),
                            )
                            rs.xadd(
                                "metrics:executor",
                                Metric(
                                    name="exec_side_depth",
                                    value=float(available_liq),
                                    labels={"symbol": sym, "side": sig.side.value},
                                ),
                            )
                            if not passed:
                                rs.xadd(
                                    "metrics:executor",
                                    Metric(name="gate_drop_reason_total", value=1.0, labels={"symbol": sym, "reason": "liquidity_insufficient"}),
                                )
                                log.info(
                                    "executor_skip_liquidity",
                                    symbol=sym,
                                    side=str(sig.side),
                                    required=required_liq,
                                    available=available_liq,
                                )
                                continue

                        signal_id = str(
                            sig.features.get("signal_id")
                            or sig.features.get("scene_id")
                            or f"{sym}:{sig.ts}"
                        )
                        requested_route = decide_route_label(
                            signal_id=signal_id,
                            rollout_mode=rollout_mode,
                            exchange_mode=exchange_mode,
                            canary_fraction=cfg.rollout.canary_fraction,
                            rng_seed=cfg.rollout.rng_seed,
                            has_real=real_exchange is not None,
                            has_paper=paper_exchange is not None,
                            default_route=default_route_label,
                        )
                        selected_exchange: ExchangeAdapter
                        if requested_route in {"real", "full", "canary"}:
                            selected_exchange = real_exchange or primary_exchange
                            if isinstance(selected_exchange, PaperExchange):
                                route_label = "paper"
                            else:
                                route_label = requested_route
                        else:
                            selected_exchange = paper_exchange or primary_exchange
                            route_label = "paper" if isinstance(selected_exchange, PaperExchange) else default_route_label

                        rs.xadd(
                            "metrics:executor",
                            Metric(name="trades_opened_total", value=1.0, labels={"symbol": sym, "route": route_label}),
                        )

                        if aggressive:
                            if sig.side == Side.BUY and ask1 > 0:
                                entry_price = min(entry_price, ask1 - tick)
                            elif sig.side == Side.SELL and bid1 > 0:
                                entry_price = max(entry_price, bid1 + tick)
                        base_step = cfg.executor.ladder.step_ticks
                        if spread_regime <= 0:
                            step_ticks = max(1, base_step // 2)
                        elif spread_regime >= 2:
                            step_ticks = max(base_step * 2, cfg.executor.passive_step_ticks)
                        else:
                            step_ticks = base_step
                        if not aggressive:
                            step_ticks = max(step_ticks, cfg.executor.passive_step_ticks)
                        else:
                            step_ticks = max(1, step_ticks)

                        ladder = build_ladder(
                            side=sig.side,
                            mid_price=entry_price,
                            qty=qty,
                            fractions=cfg.executor.ladder.fractions,
                            step_ticks=step_ticks,
                            tick_size=tick,
                            lot_step=lot,
                            min_qty=min_qty,
                            use_percent=cfg.executor.ladder.use_percent,
                        )
                        levels = []
                        for lvl_idx, (price, q) in enumerate(ladder):
                            if cfg.executor.fat_finger_ticks > 0 and abs(price - entry_price) > cfg.executor.fat_finger_ticks * tick:
                                rs.xadd(
                                    "metrics:executor",
                                    Metric(
                                        name="ladder_level_skipped_total",
                                        value=1.0,
                                        labels={"symbol": sym, "reason": "fat_finger", "level": str(lvl_idx)},
                                    ),
                                )
                                continue
                            notional = abs(price * q)
                            if caps.max_notional_per_order > 0 and notional > caps.max_notional_per_order:
                                rs.xadd(
                                    "metrics:executor",
                                    Metric(
                                        name="ladder_level_skipped_total",
                                        value=1.0,
                                        labels={"symbol": sym, "reason": "notional", "level": str(lvl_idx)},
                                    ),
                                )
                                continue
                            prob = estimate_fill_probability(sig.features, sig.side, lvl_idx, cfg.executor.fill_probability)
                            rs.xadd(
                                "metrics:executor",
                                Metric(name="expected_fill_probability", value=prob, labels={"symbol": sym, "level": str(lvl_idx)}),
                            )
                            if prob < cfg.executor.min_fill_probability:
                                rs.xadd(
                                    "metrics:executor",
                                    Metric(
                                        name="ladder_level_skipped_total",
                                        value=1.0,
                                        labels={"symbol": sym, "reason": "fill_prob", "level": str(lvl_idx)},
                                    ),
                                )
                                continue
                            bucket = probability_bucket(prob)
                            levels.append((price, q, prob, bucket))
                        if not levels:
                            rs.xadd(
                                "metrics:executor",
                                Metric(name="ladder_skip_total", value=1.0, labels={"symbol": sym, "reason": "fill_prob"}),
                            )
                            continue
                        for i, (price, q, prob, bucket) in enumerate(levels):
                            rs.xadd(
                                "metrics:executor",
                                Metric(name="exec_ladder_aggressiveness_total", value=1.0, labels={"bucket": bucket, "symbol": sym}),
                            )
                            order = selected_exchange.place_limit_post_only(
                                symbol=sym,
                                side=sig.side,
                                price=price,
                                qty=q,
                                tif=TIF[cfg.executor.tif],
                            )
                            order.scene_id = scene_id
                            open_orders.add(order.client_id)
                            order_symbol[order.client_id] = sym
                            order_exchange[order.client_id] = selected_exchange
                            order_route[order.client_id] = route_label
                            open_orders_by_symbol[sym].add(order.client_id)
                            rs.xadd(
                                "metrics:executor",
                                Metric(
                                    name="order_route_assignment",
                                    value=1.0,
                                    labels={"order_id": order.client_id, "route": route_label},
                                ),
                            )
                            rs.xadd("metrics:executor", Metric(name="open_orders", value=float(len(open_orders))))
                            if isinstance(selected_exchange, PaperExchange):
                                selected_exchange.simulate_and_publish(order=order, is_last_in_ladder=(i == len(levels) - 1))
                                open_orders.discard(order.client_id)
                                open_orders_by_symbol[sym].discard(order.client_id)
                                order_symbol.pop(order.client_id, None)
                                order_exchange.pop(order.client_id, None)
                                order_route.pop(order.client_id, None)
                                rs.xadd("metrics:executor", Metric(name="open_orders", value=float(len(open_orders))))
                            else:
                                time.sleep(cfg.executor.cancel_timeout_ms / 1000.0)
                                selected_exchange.cancel(symbol=sym, client_id=order.client_id)
                                if order.client_id in open_orders:
                                    open_orders.discard(order.client_id)
                                    open_orders_by_symbol[sym].discard(order.client_id)
                                    order_symbol.pop(order.client_id, None)
                                    order_exchange.pop(order.client_id, None)
                                    order_route.pop(order.client_id, None)
                                    rs.xadd("metrics:executor", Metric(name="open_orders", value=float(len(open_orders))))
                        if levels:
                            last_execution_side[sym] = sig.side
                            last_execution_ts[sym] = utc_ms()
                except Exception as e:
                    log.error("executor_error", error=str(e))
                finally:
                    rs.ack(stream, group, msg_id)
                processed += 1
                if max_iters_env and processed >= max_iters_env:
                    log.info("executor_exit_dry_run", processed=processed)
                    return


if __name__ == "__main__":
    main()
