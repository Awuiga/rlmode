from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class FailSafeState:
    active_until: float = 0.0


class FailSafeGuard:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.state = FailSafeState()

    def activate(self) -> None:
        if not getattr(self.cfg, "enabled", False):
            return
        duration = max(int(getattr(self.cfg, "duration_sec", 0)), 0)
        if duration <= 0:
            return
        self.state.active_until = time.monotonic() + duration

    def is_active(self) -> bool:
        if self.state.active_until <= 0:
            return False
        if time.monotonic() >= self.state.active_until:
            self.state.active_until = 0.0
            return False
        return True

    def remaining(self) -> float:
        if not self.is_active():
            return 0.0
        return max(self.state.active_until - time.monotonic(), 0.0)
