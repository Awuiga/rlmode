from __future__ import annotations

import json
import os
import shutil
from collections import deque, defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import yaml

from ..common.config import load_app_config
from ..common.logging import setup_logging, get_logger
from ..common.redis_stream import RedisStream
from ..common.schema import Fill, Metric, ControlEvent
from ..common.utils import utc_ms
from ..executor.positions import Position
from .backfill import BackfillScanner


log = get_logger("risk")


ORDER_ROUTE_TTL_MS = 60 * 60 * 1000
ORDER_ROUTE_MAX = 10000


@dataclass
class RollbackResult:
    executed: bool
    reason: str
    active_model: Optional[str]
    target_model: Optional[str]
    from_mode: str
    to_mode: str
    cooldown_until: int
    skipped: Optional[str] = None


@dataclass
class ModelInfo:
    model_id: str
    path: Path
    metadata: Dict[str, Any]


class OnlineBenchmark:
    """Track rolling winrates for baseline and canary routes."""

    def __init__(self, window: int | None):
        self._window_len: int | None = window if window and window > 0 else None

        def make_window() -> Deque[int]:
            maxlen = self._window_len if self._window_len else None
            return deque(maxlen=maxlen)

        self._samples: Dict[str, Deque[int]] = defaultdict(make_window)

    def record(self, route: str, is_win: bool) -> None:
        window = self._samples[route]
        window.append(1 if is_win else 0)

    def winrate(self, route: str) -> Tuple[float, int]:
        window = self._samples.get(route)
        if not window:
            return 0.0, 0
        sample = len(window)
        if sample == 0:
            return 0.0, 0
        return sum(window) / sample, sample

    def baseline(self) -> float:
        wins = 0
        sample = 0
        for route in ("real", "full"):
            window = self._samples.get(route)
            if not window:
                continue
            wins += sum(window)
            sample += len(window)
        if sample == 0:
            return 0.0
        return wins / sample


class RollbackManager:
    def __init__(
        self,
        *,
        guard_cfg,
        config_path: Path,
        registry_root: Path,
        initial_mode: str,
    ):
        self.guard_cfg = guard_cfg
        self.config_path = config_path
        self.registry_root = registry_root
        self.manifest_path = registry_root / "manifest.json"
        self.current_rollout_mode = initial_mode
        self.cooldown_until = 0
        self._cooldown_ms = int(max(guard_cfg.cooldown_minutes, 0) * 60 * 1000)
        self._last_parity_totals: Dict[str, float] = {}
        self._psi_windows: Dict[str, Deque[Tuple[int, float]]] = defaultdict(deque)

    def update_winrates(self, baseline: float, canary: float, sample: int, now_ms: int) -> Optional[RollbackResult]:
        if sample < int(self.guard_cfg.min_sample):
            return None
        delta_threshold = float(self.guard_cfg.winrate_delta_pp) / 100.0
        if baseline - canary > delta_threshold:
            reason = f"winrate_delta>{self.guard_cfg.winrate_delta_pp}pp"
            return self._execute(reason, now_ms)
        return None

    def record_parity_metric(self, value: float, labels: Optional[Dict[str, str]], now_ms: int) -> Optional[RollbackResult]:
        stage = (labels or {}).get("stage") or "unknown"
        key = stage
        prev = self._last_parity_totals.get(key, 0.0)
        self._last_parity_totals[key] = value
        if value > prev:
            reason = f"onnx_parity_fail(stage={stage})"
            return self._execute(reason, now_ms)
        return None

    def record_psi_metric(self, name: str, value: float, now_ms: int) -> Optional[RollbackResult]:
        window = self._psi_windows[name]
        window.append((now_ms, value))
        cutoff = now_ms - 30 * 60 * 1000
        while window and window[0][0] < cutoff:
            window.popleft()
        max_value = max((val for _, val in window), default=0.0)
        if max_value > float(self.guard_cfg.psi_block):
            reason = f"{name}>{self.guard_cfg.psi_block}"
            return self._execute(reason, now_ms)
        return None

    def record_ks_metric(self, name: str, value: float, now_ms: int) -> Optional[RollbackResult]:
        if value > float(self.guard_cfg.ks_block):
            reason = f"{name}>{self.guard_cfg.ks_block}"
            return self._execute(reason, now_ms)
        return None

    def _execute(self, reason: str, now_ms: int) -> RollbackResult:
        if now_ms < self.cooldown_until:
            return RollbackResult(
                executed=False,
                reason=reason,
                active_model=None,
                target_model=None,
                from_mode=self.current_rollout_mode,
                to_mode=self.current_rollout_mode,
                cooldown_until=self.cooldown_until,
                skipped="cooldown",
            )

        active_info = self._current_active_model()
        if active_info is None:
            return RollbackResult(
                executed=False,
                reason=reason,
                active_model=None,
                target_model=None,
                from_mode=self.current_rollout_mode,
                to_mode=self.current_rollout_mode,
                cooldown_until=self.cooldown_until,
                skipped="no_active_model",
            )

        previous_info = self._previous_model(active_info.model_id)
        if previous_info is None:
            return RollbackResult(
                executed=False,
                reason=reason,
                active_model=active_info.model_id,
                target_model=None,
                from_mode=self.current_rollout_mode,
                to_mode=self.current_rollout_mode,
                cooldown_until=self.cooldown_until,
                skipped="no_previous_model",
            )

        try:
            self._switch_active(previous_info.path)
        except Exception as exc:
            log.error("rollback_switch_failed", error=str(exc))
            return RollbackResult(
                executed=False,
                reason=reason,
                active_model=active_info.model_id,
                target_model=previous_info.model_id,
                from_mode=self.current_rollout_mode,
                to_mode=self.current_rollout_mode,
                cooldown_until=self.cooldown_until,
                skipped="switch_failed",
            )

        try:
            self._disable_rollout_mode()
        except Exception as exc:
            log.error("rollback_config_update_failed", error=str(exc))
            # proceed but note failure

        if self._cooldown_ms > 0:
            self.cooldown_until = now_ms + self._cooldown_ms
        else:
            self.cooldown_until = now_ms
        previous_mode = self.current_rollout_mode
        self.current_rollout_mode = "disabled"

        return RollbackResult(
            executed=True,
            reason=reason,
            active_model=active_info.model_id,
            target_model=previous_info.model_id,
            from_mode=previous_mode,
            to_mode="disabled",
            cooldown_until=self.cooldown_until,
        )

    def _current_active_model(self) -> Optional[ModelInfo]:
        active_path = self.registry_root / "active"
        if not active_path.exists():
            return None
        try:
            resolved = active_path.resolve(strict=False)
        except Exception:
            resolved = active_path
        meta_path = resolved / "model.json"
        if not meta_path.exists():
            return None
        try:
            metadata = json.loads(meta_path.read_text())
        except Exception:
            return None
        model_id = str(metadata.get("id") or resolved.name)
        return ModelInfo(model_id=model_id, path=resolved, metadata=metadata)

    def _previous_model(self, active_model_id: str) -> Optional[ModelInfo]:
        manifest = self._load_manifest()
        if not manifest:
            return None
        entries = sorted(
            manifest,
            key=lambda entry: str(entry.get("created_at") or ""),
        )
        active_idx = None
        for idx, entry in enumerate(entries):
            if str(entry.get("id")) == str(active_model_id):
                active_idx = idx
                break
        if active_idx is None or active_idx == 0:
            return None
        candidate = entries[active_idx - 1]
        model_path_value = candidate.get("path") or candidate.get("id")
        if model_path_value is None:
            return None
        candidate_path = Path(str(model_path_value))
        if not candidate_path.is_absolute():
            candidate_path = self.registry_root / candidate_path
        meta_path = candidate_path / "model.json"
        if not meta_path.exists():
            return None
        try:
            metadata = json.loads(meta_path.read_text())
        except Exception:
            return None
        model_id = str(metadata.get("id") or candidate_path.name)
        return ModelInfo(model_id=model_id, path=candidate_path, metadata=metadata)

    def _load_manifest(self) -> Optional[list[Dict[str, Any]]]:
        if not self.manifest_path.exists():
            return None
        try:
            data = json.loads(self.manifest_path.read_text())
        except Exception as exc:
            log.error("manifest_read_failed", error=str(exc))
            return None
        if isinstance(data, list):
            return data
        return None

    def _switch_active(self, target_dir: Path) -> None:
        active_path = self.registry_root / "active"
        if active_path.exists() or active_path.is_symlink():
            if active_path.is_symlink() or active_path.is_file():
                active_path.unlink()
            else:
                shutil.rmtree(active_path)
        try:
            active_path.symlink_to(target_dir, target_is_directory=True)
        except OSError:
            shutil.copytree(target_dir, active_path)

    def _disable_rollout_mode(self) -> None:
        cfg_path = self.config_path
        if not cfg_path.exists():
            return
        try:
            raw = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            raw = {}
        rollout = raw.get("rollout") or {}
        rollout["mode"] = "disabled"
        raw["rollout"] = rollout
        cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False))

def main():
    setup_logging()
    if os.environ.get("RL_MODE_TEST_ENTRYPOINT") == "1":
        log.info("entrypoint_test_skip", service="risk")
        return
    cfg = load_app_config()
    rs = RedisStream(cfg.redis.url, default_maxlen=cfg.redis.streams_maxlen)
    config_path = Path(os.environ.get("APP_CONFIG", "config/app.yml"))
    registry_root = Path(os.environ.get("MODEL_REGISTRY_ROOT", "models/registry"))

    rollback_manager = RollbackManager(
        guard_cfg=cfg.rollout_guard,
        config_path=config_path,
        registry_root=registry_root,
        initial_mode=(cfg.rollout.mode or "disabled").lower(),
    )

    rollout_mode = rollback_manager.current_rollout_mode
    exchange_mode = (cfg.exchange.mode or "paper").lower()

    with open("config/risk.yml", "r", encoding="utf-8") as f:
        risk_raw = yaml.safe_load(f) or {}

    risk_settings = cfg.risk.model_copy(update=risk_raw)
    daily_loss_limit = float(risk_settings.daily_loss_limit)
    max_consecutive_losses = int(risk_settings.max_consecutive_losses)
    precision_cfg = risk_settings.precision
    precision_window_len = int(precision_cfg.window)
    precision_window = deque(maxlen=precision_window_len or None)
    precision_min = float(precision_cfg.min_precision)
    precision_consecutive = int(precision_cfg.consecutive_windows)
    markout_window_len = int(risk_settings.markout.window)
    markout_window = deque(maxlen=markout_window_len or None)
    markout_short_window = deque(maxlen=5)
    markout_min = float(risk_settings.markout.min_markout)
    vol_cfg = risk_settings.volatility
    vol_enabled = bool(vol_cfg.enabled)
    vol_max_consecutive = int(vol_cfg.max_consecutive)
    vol_precision_floor = float(vol_cfg.precision_floor)
    vol_debounce_ms = int(vol_cfg.debounce_ms)
    trade_cfg = risk_settings.trade_cluster
    trade_enabled = bool(trade_cfg.enabled)
    trade_window_ms = int(trade_cfg.window_ms)
    trade_max_trades = int(trade_cfg.max_trades)
    trade_cumulative_loss = float(trade_cfg.cumulative_loss)
    trade_slo_cfg = risk_settings.trade_slo
    trade_slo_window_ms = int(trade_slo_cfg.window_seconds * 1000)
    rs.xadd(
        "metrics:risk",
        Metric(name="trades_per_hour_prev_release_p95", value=float(trade_slo_cfg.p95_prev_release)),
    )
    rs.xadd(
        "metrics:risk",
        Metric(name="p95_prev_release", value=float(trade_slo_cfg.p95_prev_release)),
    )
    exposure_cfg = risk_settings.exposure
    fail_safe_cfg = risk_settings.fail_safe

    daily_pnl = 0.0
    consecutive_losses = 0
    trades_total = 0
    wins_total = 0
    maker_total = 0
    markout_avg = 0.0
    volatility_counter = 0
    precision_breach = 0
    last_precision_val = 1.0
    last_vol_metric_ts = 0
    last_vol_halt_ts = 0
    trade_timestamps: Deque[int] = deque()
    trade_clusters: Dict[str, Deque[Tuple[int, float]]] = defaultdict(deque)
    positions: Dict[str, Position] = defaultdict(Position)

    def _make_route_stats() -> Dict[str, Any]:
        return {
            "trades_total": 0,
            "maker_total": 0,
            "markout_avg": 0.0,
            "cancel_total": 0,
            "precision_window": deque(maxlen=precision_window_len or None),
            "markout_window": deque(maxlen=markout_window_len or None),
            "markout_short_window": deque(maxlen=5),
        }

    route_stats: Dict[str, Dict[str, Any]] = defaultdict(_make_route_stats)
    order_route_map: "OrderedDict[str, tuple[str, str, int]]" = OrderedDict()

    def prune_order_routes(now_ms: int) -> None:
        cutoff = now_ms - ORDER_ROUTE_TTL_MS
        keys_to_remove = []
        for order_id, (_, _, ts_ms) in list(order_route_map.items()):
            if ts_ms < cutoff:
                keys_to_remove.append(order_id)
        for key in keys_to_remove:
            order_route_map.pop(key, None)
        while len(order_route_map) > ORDER_ROUTE_MAX:
            order_route_map.popitem(last=False)

    def record_order_route(order_id: str, route: str, symbol: str) -> None:
        if not order_id:
            return
        order_route_map[order_id] = (route, symbol, utc_ms())
        prune_order_routes(utc_ms())

    def release_route(order_id: str) -> None:
        if order_id:
            order_route_map.pop(order_id, None)

    def release_by_symbol(route: str, symbol: str | None) -> None:
        if not symbol:
            return
        for key, value in list(order_route_map.items()):
            route_val, symbol_val, _ = value
            if route_val == route and symbol_val == symbol:
                order_route_map.pop(key, None)
                break

    def emit_cancel_ratio(stats_map: Dict[str, Dict[str, Any]], stream: RedisStream, route: Optional[str]) -> None:
        if route:
            stats = stats_map[route]
            trades = int(stats["trades_total"]) or 0
            cancels = int(stats["cancel_total"]) or 0
            ratio = float(cancels) / trades if trades > 0 else 0.0
            stream.xadd(
                "metrics:risk",
                Metric(name="cancel_to_fill_ratio", value=ratio, labels={"route": route}),
            )
            return
        total_trades = sum(int(s["trades_total"]) for s in stats_map.values())
        total_cancels = sum(int(s["cancel_total"]) for s in stats_map.values())
        ratio = float(total_cancels) / total_trades if total_trades > 0 else 0.0
        stream.xadd("metrics:risk", Metric(name="cancel_to_fill_ratio", value=ratio))

    def route_for_order(order_id: str) -> str:
        entry = order_route_map.get(order_id)
        if entry:
            return entry[0]
        if order_id.startswith("paper_"):
            return "paper"
        if exchange_mode == "real":
            return "full"
        return "paper"

    def handle_rollback_result(result: Optional[RollbackResult]) -> None:
        nonlocal rollout_mode
        if not result or not result.executed:
            return
        labels = {"from": result.from_mode, "to": result.to_mode, "reason": result.reason}
        rs.xadd("metrics:risk", Metric(name="mode_switch_total", value=1.0, labels=labels))
        rs.xadd(
            "metrics:risk",
            Metric(name="rollback_total", value=1.0, labels={"reason": result.reason}),
        )
        rs.xadd(
            "control:events",
            ControlEvent(
                ts=utc_ms(),
                type="MODE_SWITCH",
                reason=result.reason,
                details={
                    "from": result.from_mode,
                    "to": result.to_mode,
                    "active_model": result.active_model,
                    "target_model": result.target_model,
                    "cooldown_until_ms": result.cooldown_until,
                },
            ),
        )
        log.warning(
            "rollout_guard_trigger",
            reason=result.reason,
            active_model=result.active_model,
            target_model=result.target_model,
        )
        rollout_mode = rollback_manager.current_rollout_mode
        cfg.rollout.mode = rollout_mode

    benchmark = OnlineBenchmark(precision_window_len)

    log.info("risk_start")
    backfill_scanner = BackfillScanner(cfg.risk.backfill_scanner, rs)

    group = "risk"
    consumer = "c1"
    streams = ["exec:fills", "metrics:signal", "metrics:executor", "metrics:ai"]

    max_iters_env = int(os.environ.get("DRY_RUN_MAX_ITER", "0")) if "DRY_RUN_MAX_ITER" in os.environ else None
    processed = 0
    while True:
        msgs = rs.read_group(group=group, consumer=consumer, streams=streams, block_ms=1000, count=100)
        if not msgs:
            backfill_scanner.maybe_scan()
            continue
        for stream, items in msgs:
            for msg_id, data in items:
                try:
                    backfill_scanner.maybe_scan()
                    if stream == "metrics:signal":
                        metric = Metric.model_validate(data)
                        if metric.name == "gate_drop" and metric.labels:
                            reason_label = metric.labels.get("reason")
                            now_ms = utc_ms()
                            if reason_label == "volatility":
                                if now_ms - last_vol_metric_ts > vol_debounce_ms:
                                    volatility_counter = 0
                                volatility_counter += 1
                                last_vol_metric_ts = now_ms
                                if (
                                    vol_enabled
                                    and volatility_counter >= vol_max_consecutive
                                    and last_precision_val <= vol_precision_floor
                                    and now_ms - last_vol_halt_ts >= vol_debounce_ms
                                ):
                                    reason = f"volatility_precision<{vol_precision_floor:.2f}"
                                    evt = ControlEvent(ts=utc_ms(), type="STOP", reason=reason)
                                    rs.xadd("control:events", evt)
                                    rs.xadd(
                                        "metrics:risk",
                                        Metric(name="kill_switch", value=1.0, labels={"reason": "volatility"}),
                                    )
                                    log.warning("risk_volatility_halt", reason=reason)
                                    last_vol_halt_ts = now_ms
                                    volatility_counter = 0
                            else:
                                volatility_counter = 0
                        continue

                    if stream == "metrics:executor":
                        metric = Metric.model_validate(data)
                        if metric.name == "order_route_assignment" and metric.labels:
                            order_id = metric.labels.get("order_id")
                            route = metric.labels.get("route")
                            symbol = metric.labels.get("symbol") if metric.labels else None
                            if order_id and route:
                                record_order_route(order_id, route, symbol or "")
                        elif metric.name == "orders_cancelled_total":
                            labels = metric.labels or {}
                            route = labels.get("route") or "paper"
                            symbol = labels.get("symbol")
                            stats = route_stats[route]
                            stats["cancel_total"] += int(metric.value)
                            emit_cancel_ratio(route_stats, rs, route)
                            emit_cancel_ratio(route_stats, rs, None)
                            release_by_symbol(route, symbol)
                        continue

                    if stream == "metrics:ai":
                        metric = Metric.model_validate(data)
                        now_metric = utc_ms()
                        if metric.name == "onnx_parity_fail_total":
                            handle_rollback_result(
                                rollback_manager.record_parity_metric(metric.value, metric.labels, now_metric)
                            )
                        elif metric.name.startswith("psi_feature_"):
                            handle_rollback_result(
                                rollback_manager.record_psi_metric(metric.name, metric.value, now_metric)
                            )
                        elif metric.name.startswith("ks_feature_"):
                            handle_rollback_result(
                                rollback_manager.record_ks_metric(metric.name, metric.value, now_metric)
                            )
                        continue

                    fill = Fill.model_validate(data)
                    positions[fill.symbol].apply_fill(fill.side.value if hasattr(fill.side, "value") else str(fill.side), fill.price, fill.qty)
                    trade_timestamps.append(fill.ts)
                    cutoff_slo = fill.ts - trade_slo_window_ms
                    while trade_timestamps and trade_timestamps[0] < cutoff_slo:
                        trade_timestamps.popleft()
                    if trade_slo_cfg.window_seconds > 0:
                        trades_per_hour = (len(trade_timestamps) * 3600) / trade_slo_cfg.window_seconds
                    else:
                        trades_per_hour = float(len(trade_timestamps))
                    rs.xadd("metrics:risk", Metric(name="trades_per_hour", value=float(trades_per_hour)))
                    trade_pnl = fill.pnl
                    daily_pnl += trade_pnl
                    trades_total += 1
                    if trade_pnl > 0:
                        wins_total += 1
                        consecutive_losses = 0
                        precision_window.append(1.0)
                    else:
                        consecutive_losses += 1
                        precision_window.append(0.0)
                    markout_window.append(fill.markout)
                    markout_short_window.append(fill.markout)

                    route_label = route_for_order(fill.order_id or "")
                    stats = route_stats[route_label]
                    stats["trades_total"] += 1
                    stats_precision: Deque[float] = stats["precision_window"]
                    stats_precision.append(1.0 if trade_pnl > 0 else 0.0)
                    route_winrate = sum(stats_precision) / (len(stats_precision) or 1)

                    stats_markout_window: Deque[float] = stats["markout_window"]
                    stats_markout_window.append(fill.markout)
                    route_markout_500 = sum(stats_markout_window) / (len(stats_markout_window) or 1)

                    stats_markout_short: Deque[float] = stats["markout_short_window"]
                    stats_markout_short.append(fill.markout)
                    route_markout_100 = sum(stats_markout_short) / (len(stats_markout_short) or 1)

                    trades_for_route = stats["trades_total"]
                    stats["markout_avg"] = (
                        (stats["markout_avg"] * (trades_for_route - 1)) + fill.markout
                    ) / trades_for_route
                    route_markout_1000 = stats["markout_avg"]

                    if fill.is_maker:
                        maker_total += 1
                        stats["maker_total"] += 1

                    markout_avg = ((markout_avg * (trades_total - 1)) + fill.markout) / trades_total
                    precision_val = sum(precision_window) / (len(precision_window) or 1)
                    markout_recent = sum(markout_window) / (len(markout_window) or 1)
                    markout_short = sum(markout_short_window) / (len(markout_short_window) or 1)
                    last_precision_val = precision_val
                    if precision_window.maxlen and len(precision_window) >= precision_window.maxlen:
                        if precision_val < precision_min:
                            precision_breach += 1
                        else:
                            precision_breach = 0
                    else:
                        precision_breach = 0

                    active_positions = sum(1 for pos in positions.values() if abs(pos.qty) > 0)
                    rs.xadd("metrics:risk", Metric(name="open_positions_total", value=float(active_positions)))
                    rs.xadd(
                        "metrics:risk",
                        Metric(name="symbol_exposure_abs", value=abs(positions[fill.symbol].qty), labels={"symbol": fill.symbol}),
                    )
                    rs.xadd("metrics:risk", Metric(name="daily_pnl", value=daily_pnl))
                    rs.xadd("metrics:risk", Metric(name="trades_total", value=1.0))
                    if trade_pnl > 0:
                        rs.xadd(
                            "metrics:risk",
                            Metric(
                                name="trades_won_total",
                                value=1.0,
                                labels={"symbol": fill.symbol, "route": route_label},
                            ),
                        )
                        rs.xadd(
                            "metrics:risk",
                            Metric(name="trades_won_total", value=1.0, labels={"route": route_label}),
                        )
                    rs.xadd("metrics:risk", Metric(name="rolling_precision", value=precision_val))
                    rs.xadd("metrics:risk", Metric(name="precision_breach_windows", value=float(precision_breach)))
                    rs.xadd("metrics:risk", Metric(name="rolling_recall", value=precision_val))
                    route_fill_rate = stats["maker_total"] / trades_for_route if trades_for_route else 0.0
                    rs.xadd("metrics:risk", Metric(name="fill_rate_rolling", value=(maker_total / trades_total)))
                    rs.xadd(
                        "metrics:risk",
                        Metric(
                            name="fill_rate_rolling",
                            value=route_fill_rate,
                            labels={"route": route_label},
                        ),
                    )
                    emit_cancel_ratio(route_stats, rs, route_label)
                    emit_cancel_ratio(route_stats, rs, None)
                    rs.xadd("metrics:risk", Metric(name="avg_markout_total", value=markout_avg))
                    rs.xadd("metrics:risk", Metric(name="avg_markout_500ms", value=markout_recent))
                    rs.xadd("metrics:risk", Metric(name="avg_markout_100ms", value=markout_short))
                    benchmark.record(route_label, trade_pnl > 0)
                    baseline_winrate = benchmark.baseline()
                    rs.xadd("metrics:risk", Metric(name="winrate_baseline_active", value=baseline_winrate))
                    canary_winrate, canary_sample = benchmark.winrate("canary")
                    rs.xadd("metrics:risk", Metric(name="winrate_canary", value=canary_winrate))
                    rs.xadd("metrics:risk", Metric(name="sample_canary", value=float(canary_sample)))
                    handle_rollback_result(
                        rollback_manager.update_winrates(baseline_winrate, canary_winrate, canary_sample, now_ms)
                    )
                    rs.xadd(
                        "metrics:risk",
                        Metric(name="winrate_rolling", value=route_winrate, labels={"route": route_label}),
                    )
                    rs.xadd(
                        "metrics:risk",
                        Metric(name="avg_markout_100ms", value=route_markout_100, labels={"route": route_label}),
                    )
                    rs.xadd(
                        "metrics:risk",
                        Metric(name="avg_markout_500ms", value=route_markout_500, labels={"route": route_label}),
                    )
                    rs.xadd(
                        "metrics:risk",
                        Metric(name="avg_markout_1000ms", value=route_markout_1000, labels={"route": route_label}),
                    )

                    release_route(fill.order_id or "")

                    trigger = None
                    reason = ""
                    if trade_enabled and trade_pnl < 0 and fill.scene_id:
                        cluster = trade_clusters[fill.scene_id]
                        cluster.append((fill.ts, trade_pnl))
                        cutoff_ts = fill.ts - trade_window_ms
                        while cluster and cluster[0][0] < cutoff_ts:
                            cluster.popleft()
                        cumulative_scene_loss = sum(p for _, p in cluster)
                        if len(cluster) >= trade_max_trades and cumulative_scene_loss <= trade_cumulative_loss:
                            trigger = "STOP"
                            reason = f"trade_cluster(scene={fill.scene_id})"
                            cluster.clear()
                    elif trade_enabled and fill.scene_id and trade_pnl >= 0:
                        trade_clusters.pop(fill.scene_id, None)

                    if trades_per_hour > trade_slo_cfg.max_trades:
                        rs.xadd(
                            "metrics:risk",
                            Metric(name="trade_slo_breach_total", value=1.0, labels={"symbol": fill.symbol}),
                        )
                        if trigger is None and not trade_slo_cfg.alert_only:
                            trigger = "STOP"
                            reason = f"trade_rate>{trade_slo_cfg.max_trades}"

                    if trigger is None:
                        if daily_pnl <= daily_loss_limit:
                            trigger = "STOP"
                            reason = f"daily_pnl<=limit ({daily_pnl:.4f}<={daily_loss_limit})"
                        elif consecutive_losses >= max_consecutive_losses:
                            trigger = "STOP"
                            reason = f"consecutive_losses>={max_consecutive_losses}"
                        elif precision_breach >= precision_consecutive:
                            trigger = "STOP"
                            reason = f"precision_window<{precision_min:.2f}"
                            precision_breach = 0
                        elif markout_window.maxlen and len(markout_window) >= markout_window.maxlen and markout_recent <= markout_min:
                            trigger = "STOP"
                            reason = f"markout_recent<={markout_min:.4f}"
                        elif exposure_cfg.max_positions and active_positions > exposure_cfg.max_positions:
                            trigger = "STOP"
                            reason = "exposure_max_positions"
                        elif exposure_cfg.per_symbol.get(fill.symbol) and abs(positions[fill.symbol].qty) > exposure_cfg.per_symbol[fill.symbol]:
                            trigger = "STOP"
                            reason = f"exposure_symbol>{exposure_cfg.per_symbol[fill.symbol]}"
                        elif (
                            exposure_cfg.kelly_fraction > 0
                            and exposure_cfg.reference_equity > 0
                            and abs(positions[fill.symbol].qty) * abs(fill.price) > exposure_cfg.reference_equity * exposure_cfg.kelly_fraction
                        ):
                            trigger = "STOP"
                            reason = "exposure_kelly"
                    if trigger:
                        evt = ControlEvent(ts=utc_ms(), type=trigger, reason=reason)
                        rs.xadd("control:events", evt)
                        rs.xadd("metrics:risk", Metric(name="kill_switch", value=1.0, labels={"reason": reason}))
                        if any(key in reason for key in ("precision", "trade_cluster", "volatility", "exposure", "trade_rate")):
                            rs.xadd("control:events", ControlEvent(ts=utc_ms(), type="ROLLBACK", reason=reason))
                        if fail_safe_cfg.enabled:
                            details = {"mode": fail_safe_cfg.mode, "duration_sec": fail_safe_cfg.duration_sec}
                            rs.xadd("control:events", ControlEvent(ts=utc_ms(), type="MODE_SWITCH", reason=reason, details=details))
                        log.warning("risk_halt", reason=reason)
                except Exception as e:
                    log.error("risk_error", error=str(e))
                finally:
                    rs.ack(stream, group, msg_id)
                processed += 1
                if max_iters_env and processed >= max_iters_env:
                    log.info("risk_exit_dry_run", processed=processed)
                    return


if __name__ == "__main__":
    main()
