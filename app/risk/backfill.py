from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq

from ..common.schema import Metric


@dataclass
class BackfillState:
    last_scan: float = 0.0


class BackfillScanner:
    def __init__(self, cfg, rs) -> None:
        self.cfg = cfg
        self.rs = rs
        self.state = BackfillState()

    def maybe_scan(self) -> None:
        if not getattr(self.cfg, "enabled", False):
            return
        interval = max(int(getattr(self.cfg, "check_interval_sec", 300)), 1)
        now = time.time()
        if now - self.state.last_scan < interval:
            return
        self.state.last_scan = now
        path = Path(getattr(self.cfg, "parquet_path", ""))
        if not path.exists():
            return
        ts_column = getattr(self.cfg, "timestamp_column", "ts")
        gap_threshold = max(int(getattr(self.cfg, "gap_threshold_ms", 60_000)), 1)
        try:
            dataset = pq.ParquetDataset(path)
        except Exception:  # pragma: no cover - pyarrow errors depend on FS
            return
        try:
            table = dataset.read(columns=[ts_column])
        except AttributeError:
            try:
                table = dataset.read_table(columns=[ts_column])
            except Exception:
                return
        if table.num_rows == 0:
            return
        column = table.column(0)
        timestamps = column.to_pylist()
        previous_ts: Optional[int] = None
        worst_gap = 0
        for ts in timestamps:
            try:
                ts_int = int(ts)
            except (TypeError, ValueError):
                continue
            if previous_ts is not None:
                gap = ts_int - previous_ts
                if gap > worst_gap:
                    worst_gap = gap
            previous_ts = ts_int
        if worst_gap > gap_threshold:
            self.rs.xadd(
                "metrics:risk",
                Metric(
                    name="parquet_gap_detected_ms",
                    value=float(worst_gap),
                    labels={"path": str(path)},
                ),
            )
