from __future__ import annotations

from typing import Dict, Iterable

from ..common.schema import Metric


def validate_exchange_reference(exchange, symbols_meta: Dict[str, Dict[str, float]], rs, *, stream: str = "metrics:executor") -> None:
    mismatches: list[tuple[str, str, float, float]] = []
    for symbol, meta in symbols_meta.items():
        try:
            live = exchange.get_min_steps(symbol=symbol)
        except Exception:  # pragma: no cover - network errors logged elsewhere
            continue
        if not isinstance(live, dict):
            continue
        expected_fields: Iterable[str] = {"tick_size", "lot_step", "min_qty", "mmr", "fees"}
        for field in expected_fields:
            ref_val = _safe_float(meta.get(field))
            live_val = _safe_float(live.get(field))
            if ref_val is None or live_val is None:
                continue
            if not _close(ref_val, live_val):
                mismatches.append((symbol, field, ref_val, live_val))
    for symbol, field, ref_val, live_val in mismatches:
        rs.xadd(
            stream,
            Metric(
                name="exchange_reference_mismatch_total",
                value=1.0,
                labels={"symbol": symbol, "field": field, "expected": f"{ref_val}", "actual": f"{live_val}"},
            ),
        )


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _close(lhs: float, rhs: float, *, tol: float = 1e-6) -> bool:
    return abs(lhs - rhs) <= tol * max(1.0, abs(lhs), abs(rhs))
