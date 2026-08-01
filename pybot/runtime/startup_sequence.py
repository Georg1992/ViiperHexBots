"""Per-hunt startup sequencing state.

The startup sequence is deliberately independent from the broader runtime
lifecycle gates.  It coordinates the three milestones that release combat:
area-clear discovery, character buffs, and normal skill timers.
"""

from __future__ import annotations

import threading


class HuntStartupSequence:
    """Own startup milestones and the generation of the active hunt cycle.

    The sequence starts in an unmanaged/ready state so lightweight gate
    fixtures retain their historical behavior.  Production calls ``begin``
    before workers start, which enables startup gating and clears all
    milestones.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self._managed = False
        self.area_clear = threading.Event()
        self.buffs_done = threading.Event()
        self.timers_done = threading.Event()
        # Before production opts in, startup is not a gate for fixture users.
        self.buffs_done.set()
        self.timers_done.set()

    @property
    def generation(self) -> int:
        """Return the current hunt generation."""
        with self._lock:
            return self._generation

    def begin(self) -> None:
        """Enable startup management and clear the first hunt's milestones."""
        with self._lock:
            self._managed = True
            self._clear_milestones()

    def begin_new_hunt(self) -> None:
        """Advance to a new hunt and reset its startup milestones atomically."""
        with self._lock:
            self._generation += 1
            if self._managed:
                self._clear_milestones()

    def mark_area_clear(self, clear: bool = True) -> None:
        """Record whether discovery currently sees an empty area."""
        if clear:
            self.area_clear.set()
        else:
            self.area_clear.clear()

    def mark_buffs_done(self) -> None:
        """Release the normal timer startup phase."""
        self.buffs_done.set()

    def mark_timers_done(self) -> None:
        """Release combat after this cycle's normal timers have fired."""
        self.timers_done.set()

    def is_combat_ready(self) -> bool:
        """Return whether startup no longer blocks combat."""
        return self.timers_done.is_set()

    def _clear_milestones(self) -> None:
        # Called only while _lock is held. Keeping this operation together with
        # generation changes prevents a sit-to-hunt transition from publishing
        # a new generation with stale completion events.
        self.area_clear.clear()
        self.buffs_done.clear()
        self.timers_done.clear()


__all__ = ["HuntStartupSequence"]
