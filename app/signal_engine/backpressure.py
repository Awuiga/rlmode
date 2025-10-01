from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BackpressureState:
    active: bool = False
    skip_mod: int = 2
    counter: int = 0


class BackpressureController:
    def __init__(self, cfg, redis_stream) -> None:
        self.cfg = cfg
        self.rs = redis_stream
        self.state = BackpressureState()
        self.last_check_ms = 0

    def should_throttle(self, now_ms: int) -> bool:
        if not getattr(self.cfg, "enabled", False):
            return False
        if now_ms - self.last_check_ms >= max(int(getattr(self.cfg, "check_interval_ms", 500)), 100):
            self.last_check_ms = now_ms
            pending = self.rs.pending_length(self.cfg.stream, self.cfg.group)
            backlog = max(pending, self.rs.stream_length(self.cfg.stream))
            threshold = int(getattr(self.cfg, "pending_threshold", 200) or 0)
            release = int(getattr(self.cfg, "release_threshold", threshold // 2 or 1))
            if backlog >= threshold:
                self._activate()
            elif backlog <= release:
                self._deactivate()
        if not self.state.active:
            return False
        self.state.counter = (self.state.counter + 1) % self.state.skip_mod
        return self.state.counter == 0

    def _activate(self) -> None:
        if self.state.active:
            return
        drop_rate = max(min(float(getattr(self.cfg, "drop_rate", 0.5) or 0.0), 0.95), 0.05)
        skip_mod = max(int(round(1.0 / drop_rate)), 1)
        self.state = BackpressureState(active=True, skip_mod=skip_mod)

    def _deactivate(self) -> None:
        if not self.state.active:
            return
        self.state = BackpressureState(active=False, skip_mod=self.state.skip_mod)
