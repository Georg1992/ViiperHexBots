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
        # Incremented at the moment a teleport input is accepted. Observation
        # producers capture this token before their read and may publish only
        # if it is still current. A newly-created epoch remains quarantined
        # until teleport settle completes; this prevents transition-frame OCR
        # (including the previous area's SP value) from becoming actionable.
        self._observation_epoch = 0
        self._observation_epoch_ready = True

    @property
    def observation_epoch(self) -> int:
        """Current screen-observation epoch, read atomically."""
        with self._lock:
            return self._observation_epoch

    def begin_observation_epoch(self) -> int:
        """Invalidate old samples and quarantine the landing transition.

        Producers stay alive and may finish reads, but every publication is
        rejected until :meth:`complete_observation_epoch` is called after the
        teleport settle. This is stronger than an epoch token alone: a read
        that starts after the key press can still capture the old/loading frame.
        """
        with self._lock:
            self._observation_epoch += 1
            self._observation_epoch_ready = False
            now = int(time.monotonic() * 1000)
            self._hp = None
            self._hp_max = None
            self._sp = None
            self._sp_max = None
            self._weight = None
            self._weight_max = None
            self._hp_observed_ms = now
            self._sp_observed_ms = now
            self._weight_observed_ms = now
            self._hp_changed_ms = now
            self._sp_changed_ms = now
            self._weight_changed_ms = now
            return self._observation_epoch

    def complete_observation_epoch(self, epoch: int | None = None) -> bool:
        """Open the settled epoch for fresh producer publications."""
        with self._lock:
            if epoch is not None and epoch != self._observation_epoch:
                return False
            self._observation_epoch_ready = True
            return True

    def _epoch_is_current(self, epoch: int | None) -> bool:
        return (
            (epoch is None or epoch == self._observation_epoch)
            and self._observation_epoch_ready
        )

    # ── HP ────────────────────────────────────────────────────────

    def publish_snapshot_if_current(
        self,
        hp: int | None,
        hp_max: int | None,
        sp: int | None,
        sp_max: int | None,
        weight: int | None,
        weight_max: int | None,
        epoch: int,
    ) -> bool:
        """Atomically publish a complete OCR snapshot for the current epoch."""
        with self._lock:
            if not self._epoch_is_current(epoch):
                return False
            now = int(time.monotonic() * 1000)
            self._hp_observed_ms = now
            self._sp_observed_ms = now
            self._weight_observed_ms = now
            if hp != self._hp or hp_max != self._hp_max:
                self._hp = hp
                self._hp_max = hp_max
                self._hp_changed_ms = now
            if sp != self._sp or sp_max != self._sp_max:
                self._sp = sp
                self._sp_max = sp_max
                self._sp_changed_ms = now
            if weight != self._weight or weight_max != self._weight_max:
                self._weight = weight
                self._weight_max = weight_max
                self._weight_changed_ms = now
            return True

    # ── HP ────────────────────────────────────────────────────────

    def publish_hp(self, hp: int | None, hp_max: int | None) -> None:
        with self._lock:
            now = int(time.monotonic() * 1000)
            self._hp_observed_ms = now
            if hp != self._hp or hp_max != self._hp_max:
                self._hp = hp
                self._hp_max = hp_max
                self._hp_changed_ms = now

    def publish_hp_if_current(
        self,
        hp: int | None,
        hp_max: int | None,
        epoch: int,
    ) -> bool:
        with self._lock:
            if not self._epoch_is_current(epoch):
                return False
            now = int(time.monotonic() * 1000)
            self._hp_observed_ms = now
            if hp != self._hp or hp_max != self._hp_max:
                self._hp = hp
                self._hp_max = hp_max
                self._hp_changed_ms = now
            return True

    def hp_pair(self) -> tuple[int | None, int | None]:
        """Atomic ``(hp, hp_max)`` for danger checks."""
        with self._lock:
            return self._hp, self._hp_max

    def hp_sample(self) -> tuple[int | None, int | None, int, int]:
        """Atomic ``(hp, hp_max, hp_observed_ms, hp_changed_ms)``.

        The observation clock lets healers reject a stale pre-heal reading
        instead of healing twice while the vitals publisher catches up.
        """
        with self._lock:
            return self._hp, self._hp_max, self._hp_observed_ms, self._hp_changed_ms

    # ── SP ────────────────────────────────────────────────────────

    def publish_sp(self, sp: int | None, sp_max: int | None) -> None:
        with self._lock:
            now = int(time.monotonic() * 1000)
            self._sp_observed_ms = now
            if sp != self._sp or sp_max != self._sp_max:
                self._sp = sp
                self._sp_max = sp_max
                self._sp_changed_ms = now

    def publish_sp_if_current(
        self,
        sp: int | None,
        sp_max: int | None,
        epoch: int,
    ) -> bool:
        with self._lock:
            if not self._epoch_is_current(epoch):
                return False
            now = int(time.monotonic() * 1000)
            self._sp_observed_ms = now
            if sp != self._sp or sp_max != self._sp_max:
                self._sp = sp
                self._sp_max = sp_max
                self._sp_changed_ms = now
            return True

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

    def publish_weight_if_current(
        self,
        weight: int | None,
        weight_max: int | None,
        epoch: int,
    ) -> bool:
        with self._lock:
            if not self._epoch_is_current(epoch):
                return False
            now = int(time.monotonic() * 1000)
            self._weight_observed_ms = now
            if weight != self._weight or weight_max != self._weight_max:
                self._weight = weight
                self._weight_max = weight_max
                self._weight_changed_ms = now
            return True

    def weight_pair(self) -> tuple[int | None, int | None]:
        """Atomic ``(weight, weight_max)`` for storage checks."""
        with self._lock:
            return self._weight, self._weight_max

    def clear_weight(self) -> None:
        self.publish_weight(None, None)
