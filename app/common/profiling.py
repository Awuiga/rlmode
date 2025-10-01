from __future__ import annotations

import statistics
import time
from collections import deque
from contextlib import contextmanager
from typing import Deque, Dict, Iterable

from .schema import Metric


class LatencyProfiler:
    """Track rolling latency percentiles and emit metrics periodically."""

    def __init__(
        self,
        *,
        rs,
        stream: str,
        metric_name: str,
        labels: Dict[str, str] | None = None,
        window: int = 256,
        emit_every: int = 64,
    ) -> None:
        self.rs = rs
        self.stream = stream
        self.metric_name = metric_name
        self.labels = dict(labels or {})
        self.samples: Deque[float] = deque(maxlen=max(window, 8))
        self.emit_every = max(emit_every, 1)
        self.counter = 0
        self._last_emit = 0.0

    @contextmanager
    def track(self) -> Iterable[None]:
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            end = time.perf_counter_ns()
            duration_ms = (end - start) / 1_000_000.0
            self.record(duration_ms)

    def record(self, duration_ms: float) -> None:
        self.samples.append(float(duration_ms))
        self.counter += 1
        if self.counter >= self.emit_every:
            self.counter = 0
            self.emit()

    def emit(self) -> None:
        if not self.samples:
            return
        sorted_samples = sorted(self.samples)
        p50 = _percentile(sorted_samples, 0.5)
        p95 = _percentile(sorted_samples, 0.95)
        p99 = _percentile(sorted_samples, 0.99)
        stats = {
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "mean": statistics.fmean(sorted_samples),
        }
        for quantile, value in stats.items():
            metric_labels = {"quantile": quantile, **self.labels}
            self.rs.xadd(
                self.stream,
                Metric(name=self.metric_name, value=float(value), labels=metric_labels),
            )


def _percentile(sorted_samples: list[float], quantile: float) -> float:
    if not sorted_samples:
        return 0.0
    if quantile <= 0:
        return sorted_samples[0]
    if quantile >= 1:
        return sorted_samples[-1]
    idx = quantile * (len(sorted_samples) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_samples) - 1)
    weight = idx - lower
    return sorted_samples[lower] * (1 - weight) + sorted_samples[upper] * weight
