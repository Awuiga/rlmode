from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class WarmupConfigView:
    enabled: bool = True
    seconds: int = 0


class WarmupGate:
    def __init__(self, cfg: WarmupConfigView, *, clock: Callable[[], float] | None = None) -> None:
        self.enabled = bool(getattr(cfg, "enabled", False))
        duration = max(int(getattr(cfg, "seconds", 0)), 0)
        self._clock = clock or time.monotonic
        self._deadline = self._clock() + duration if self.enabled and duration > 0 else 0.0

    def should_drop(self) -> bool:
        if not self.enabled:
            return False
        if self._deadline <= 0:
            return False
        return self._clock() < self._deadline

    def remaining(self) -> float:
        if not self.enabled or self._deadline <= 0:
            return 0.0
        return max(self._deadline - self._clock(), 0.0)


class BackpressureGuard:
    def __init__(self, cfg, *, clock: Callable[[], float] | None = None) -> None:
        self.enabled = bool(getattr(cfg, "enabled", False))
        self.max_queue_len = max(int(getattr(cfg, "max_queue_len", 0) or 0), 0)
        self.drop_mode = str(getattr(cfg, "drop_mode", "halt") or "halt").lower()
        if self.drop_mode not in {"halt", "degrade"}:
            self.drop_mode = "halt"
        self._clock = clock or time.monotonic
        self._last_transition: Optional[float] = None
        self._active_mode: Optional[str] = None

    def evaluate(self, queue_length: int) -> Optional[str]:
        if not self.enabled or self.max_queue_len <= 0:
            self._active_mode = None
            return None
        if queue_length > self.max_queue_len:
            if self._active_mode != self.drop_mode:
                self._last_transition = self._clock()
            self._active_mode = self.drop_mode
            return self.drop_mode
        if self._active_mode is not None:
            self._last_transition = self._clock()
        self._active_mode = None
        return None

    def is_degraded(self) -> bool:
        return self._active_mode == "degrade"

    def should_drop(self) -> bool:
        return self._active_mode == "halt"

    def last_transition(self) -> Optional[float]:
        return self._last_transition
