import time
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from app.common.profiling import LatencyProfiler
from app.common.schema import Metric
from app.executor.failsafe import FailSafeGuard

from app.executor.resilience import WarmupGate, BackpressureGuard

from app.risk.backfill import BackfillScanner
from app.signal_engine.backpressure import BackpressureController


class DummyStream:
    def __init__(self):
        self.records = []

    def xadd(self, stream: str, metric: Metric, idempotency_key=None):
        self.records.append((stream, metric, idempotency_key))

    def pending_length(self, stream: str, group: str) -> int:
        return 0

    def stream_length(self, stream: str) -> int:
        return 0


def test_latency_profiler_emits_metrics():
    rs = DummyStream()
    profiler = LatencyProfiler(
        rs=rs,
        stream="metrics:perf",
        metric_name="latency_ms",
        labels={"stage": "test"},
        window=16,
        emit_every=3,
    )
    for _ in range(3):
        with profiler.track():
            time.sleep(0.001)
    assert any(m.name == "latency_ms" and m.labels.get("quantile") == "p50" for _, m, _ in rs.records)



def test_warmup_gate():
    now = [0.0]

    def fake_clock():
        return now[0]

    gate = WarmupGate(SimpleNamespace(enabled=True, seconds=90), clock=fake_clock)
    assert gate.should_drop() is True
    now[0] = 91.0
    assert gate.should_drop() is False
    assert gate.remaining() == 0.0


def test_backpressure_guard_modes():
    halt_guard = BackpressureGuard(SimpleNamespace(enabled=True, max_queue_len=5, drop_mode="halt"))
    assert halt_guard.evaluate(10) == "halt"
    assert halt_guard.should_drop() is True
    assert halt_guard.evaluate(1) is None
    assert halt_guard.should_drop() is False

    degrade_guard = BackpressureGuard(SimpleNamespace(enabled=True, max_queue_len=5, drop_mode="degrade"))
    assert degrade_guard.evaluate(10) == "degrade"
    assert degrade_guard.is_degraded() is True
    assert degrade_guard.evaluate(1) is None
    assert degrade_guard.is_degraded() is False



def test_backpressure_throttles_on_backlog():
    class FakeRedis:
        def __init__(self):
            self.pending = 0
            self.length = 0

        def pending_length(self, stream, group):
            return self.pending

        def stream_length(self, stream):
            return self.length

    cfg = SimpleNamespace(
        enabled=True,
        stream="sig:candidates",
        group="aiscore",
        pending_threshold=10,
        release_threshold=5,
        drop_rate=0.5,
        check_interval_ms=0,
    )
    fake = FakeRedis()
    ctrl = BackpressureController(cfg, fake)
    fake.pending = 20
    # Controller arms under sustained backlog and drops periodically while active
    dropped = sum(ctrl.should_throttle((i + 1) * 200) for i in range(6))
    assert dropped >= 1



def test_fail_safe_trigger(monkeypatch):

    now = [0.0]

    def fake_monotonic():
        return now[0]


    guard = FailSafeGuard(SimpleNamespace(enabled=True, duration_sec=60), clock=fake_monotonic)
    assert guard.activate(reason="crit") is True
    assert guard.is_active() is True
    assert guard.remaining() == 60
    assert guard.activate(reason="crit") is False  # already active, extend window
    now[0] = 61.0
    assert guard.is_active() is False
    assert guard.remaining() == 0.0



def test_backfill_scanner_detects_gap(tmp_path):
    table1 = pa.table({"ts": np.array([0, 1_000], dtype=np.int64)})
    table2 = pa.table({"ts": np.array([10_000, 12_000], dtype=np.int64)})
    pq.write_table(table1, tmp_path / "part1.parquet")
    pq.write_table(table2, tmp_path / "part2.parquet")
    rs = DummyStream()
    cfg = SimpleNamespace(
        enabled=True,
        parquet_path=str(tmp_path),
        timestamp_column="ts",
        gap_threshold_ms=5_000,
        check_interval_sec=0,
    )
    scanner = BackfillScanner(cfg, rs)
    scanner.maybe_scan()
    assert any(m.name == "parquet_gap_detected_ms" for _, m, _ in rs.records)
