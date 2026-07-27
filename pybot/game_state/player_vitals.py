"""Shared final player vitals (SP) for UI publishers and hunt workers.

UI polls (memory or status-panel OCR) publish into this store. Workers only
read the final values — they must not re-OCR or re-read process memory.
"""

from __future__ import annotations

import threading
import time


class PlayerVitals:
    """Thread-safe holder for the latest known SP / SP max."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sp: int | None = None
        self._sp_max: int | None = None
        self._updated_ms: int = 0

    def publish_sp(self, sp: int | None, sp_max: int | None) -> None:
        with self._lock:
            self._sp = sp
            self._sp_max = sp_max
            self._updated_ms = int(time.monotonic() * 1000)

    def clear_sp(self) -> None:
        self.publish_sp(None, None)

    @property
    def sp(self) -> int | None:
        with self._lock:
            return self._sp

    @property
    def sp_max(self) -> int | None:
        with self._lock:
            return self._sp_max

    @property
    def updated_ms(self) -> int:
        with self._lock:
            return self._updated_ms

    def sp_pair(self) -> tuple[int | None, int | None]:
        """Atomic ``(sp, sp_max)`` for ratio checks."""
        with self._lock:
            return self._sp, self._sp_max

    def sp_sample(self) -> tuple[int | None, int]:
        """Atomic ``(sp, updated_ms)`` for pre/post idle comparisons."""
        with self._lock:
            return self._sp, self._updated_ms
