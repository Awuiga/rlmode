from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class FailSafeState:
    active_until: float = 0.0
    last_reason: Optional[str] = None


class FailSafeGuard:
    def __init__(self, cfg, *, clock: Callable[[], float] | None = None) -> None:
        self.cfg = cfg
        self.state = FailSafeState()
        self._clock = clock or time.monotonic

    def _duration(self) -> int:
        return max(int(getattr(self.cfg, "duration_sec", 0)), 0)

    def activate(self, *, reason: Optional[str] = None, duration_override: Optional[int] = None) -> bool:
        if not getattr(self.cfg, "enabled", False):
            return False
        duration = self._duration() if duration_override is None else max(int(duration_override), 0)
        if duration <= 0:
            return False
        now = self._clock()
        was_active = now < self.state.active_until
        deadline = now + duration
        if was_active:
            self.state.active_until = max(self.state.active_until, deadline)
        else:
            self.state.active_until = deadline
        self.state.last_reason = reason
        return not was_active

    def is_active(self) -> bool:
        if self.state.active_until <= 0:
            return False
        if self._clock() >= self.state.active_until:
            self.state.active_until = 0.0
            self.state.last_reason = None
            return False
        return True

    def remaining(self) -> float:
        if not self.is_active():
            return 0.0
        return max(self.state.active_until - self._clock(), 0.0)

    def last_reason(self) -> Optional[str]:
        return self.state.last_reason
