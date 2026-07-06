"""Post-attack direct state queue — mirrors HuntScheduleDirectStateCheck."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class UrgentDirectStateRequest:
    track_id: int
    x: int
    y: int
    ready_at_ms: int


class UrgentStateQueue:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: UrgentDirectStateRequest | None = None

    def schedule_direct(
        self,
        track_id: int,
        x: int,
        y: int,
        *,
        delay_ms: int,
        now_ms: int,
    ) -> None:
        with self._lock:
            self._pending = UrgentDirectStateRequest(
                track_id=track_id,
                x=x,
                y=y,
                ready_at_ms=now_ms + delay_ms,
            )

    def has_pending(self) -> bool:
        with self._lock:
            return self._pending is not None

    def pop_ready(self, now_ms: int) -> UrgentDirectStateRequest | None:
        with self._lock:
            if self._pending is None:
                return None
            if now_ms < self._pending.ready_at_ms:
                return None
            request = self._pending
            self._pending = None
            return request

    def clear(self) -> None:
        with self._lock:
            self._pending = None
