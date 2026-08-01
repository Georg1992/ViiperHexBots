"""Per-hunt startup sequencing state.

The startup sequence is deliberately independent from the broader runtime
lifecycle gates. It tracks area-clear discovery and character buffs before
safe startup actions; normal skill timers have a separate auxiliary milestone.
"""

from __future__ import annotations

import threading


class HuntStartupSequence:
    """Own startup milestones and the generation of the active hunt cycle.

    The sequence starts in an unmanaged/ready state so lightweight gate
    fixtures retain their historical behavior. Production calls ``begin``
    before workers start, which enables startup gating. Initial combat remains
    available while discovery is finding/clearing the area; once the area is
    confirmed clear, combat waits for character buffs. Normal skill timers are
    auxiliary and never hold the attack loop hostage.
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
        """Record that the safe character-buff phase has completed."""
        self.buffs_done.set()

    def mark_timers_done(self) -> None:
        """Record that this cycle's auxiliary timers have fired once."""
        self.timers_done.set()

    def is_combat_ready(self) -> bool:
        """Return whether the character-buff phase no longer blocks combat.

        Normal skill timers are auxiliary work. They may be staggered at hunt
        startup, but one slow/missing timer must never prevent the attack loop
        from starting after the area is clear and the safe character-buff
        phase is complete.
        """
        # The first scan may find mobs. Combat must remain available so the
        # hunt can clear that area; only the confirmed-clear phase is gated on
        # completion of safe character buffs. Unmanaged fixtures stay ready.
        return (
            not self._managed
            or not self.area_clear.is_set()
            or self.buffs_done.is_set()
        )

    def _clear_milestones(self) -> None:
        # Called only while _lock is held. Keeping this operation together with
        # generation changes prevents a sit-to-hunt transition from publishing
        # a new generation with stale completion events.
        self.area_clear.clear()
        self.buffs_done.clear()
        self.timers_done.clear()


__all__ = ["HuntStartupSequence"]
