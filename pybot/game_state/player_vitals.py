"""Shared final player vitals for UI publishers and hunt workers.

UI polls (memory or status-panel OCR) publish into this store. Workers only
read the final values — they must not re-OCR or re-read process memory.
"""

from __future__ import annotations

import threading
import time


class PlayerVitals:
    """Thread-safe holder for the latest known HP, SP and Weight values."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hp: int | None = None
        self._hp_max: int | None = None
        self._sp: int | None = None
        self._sp_max: int | None = None
        self._weight: int | None = None
        self._weight_max: int | None = None
        # Per-vital clocks so HP/weight publishes cannot fake SP freshness.
        self._hp_observed_ms: int = 0
        self._hp_changed_ms: int = 0
        self._sp_observed_ms: int = 0
        self._sp_changed_ms: int = 0
        self._weight_observed_ms: int = 0
        self._weight_changed_ms: int = 0

    # ── HP ────────────────────────────────────────────────────────

    def publish_hp(self, hp: int | None, hp_max: int | None) -> None:
        with self._lock:
            now = int(time.monotonic() * 1000)
            self._hp_observed_ms = now
            if hp != self._hp or hp_max != self._hp_max:
                self._hp = hp
                self._hp_max = hp_max
                self._hp_changed_ms = now

    def hp_pair(self) -> tuple[int | None, int | None]:
        """Atomic ``(hp, hp_max)`` for danger checks."""
        with self._lock:
            return self._hp, self._hp_max

    # ── SP ────────────────────────────────────────────────────────

    def publish_sp(self, sp: int | None, sp_max: int | None) -> None:
        with self._lock:
            now = int(time.monotonic() * 1000)
            self._sp_observed_ms = now
            if sp != self._sp or sp_max != self._sp_max:
                self._sp = sp
                self._sp_max = sp_max
                self._sp_changed_ms = now

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
        """Last SP observation time (compat alias for ``observed_ms``)."""
        with self._lock:
            return self._sp_observed_ms

    @property
    def observed_ms(self) -> int:
        """Last SP observation time."""
        with self._lock:
            return self._sp_observed_ms

    @property
    def changed_ms(self) -> int:
        """Last SP value-change time."""
        with self._lock:
            return self._sp_changed_ms

    def sp_pair(self) -> tuple[int | None, int | None]:
        """Atomic ``(sp, sp_max)`` for ratio checks."""
        with self._lock:
            return self._sp, self._sp_max

    def sp_sample(self) -> tuple[int | None, int, int]:
        """Atomic ``(sp, sp_observed_ms, sp_changed_ms)`` for idle comparisons."""
        with self._lock:
            return self._sp, self._sp_observed_ms, self._sp_changed_ms

    # ── Weight ────────────────────────────────────────────────────

    def publish_weight(self, weight: int | None, weight_max: int | None) -> None:
        with self._lock:
            now = int(time.monotonic() * 1000)
            self._weight_observed_ms = now
            if weight != self._weight or weight_max != self._weight_max:
                self._weight = weight
                self._weight_max = weight_max
                self._weight_changed_ms = now

    def weight_pair(self) -> tuple[int | None, int | None]:
        """Atomic ``(weight, weight_max)`` for storage checks."""
        with self._lock:
            return self._weight, self._weight_max

    def clear_weight(self) -> None:
        self.publish_weight(None, None)
