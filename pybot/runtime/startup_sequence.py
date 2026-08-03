"""Per-hunt startup sequencing state.

The startup sequence is deliberately independent from the broader runtime
lifecycle gates. It tracks area-clear discovery, character buffs, and timer
startup casts before combat is released.

A fresh hunt is trusted to start at a safe location, so the initial
area-clear milestone starts satisfied and startup buffs/timers run
immediately without waiting for the first discovery scan. Recovered hunts
(sit/stand, teleport) re-arm the area-clear wait so the bot confirms an
empty landing area before casting again.
"""

from __future__ import annotations

import threading


class HuntStartupSequence:
    """Own startup milestones and the generation of the active hunt cycle.

    The sequence starts in an unmanaged/ready state so lightweight gate
    fixtures retain their historical behavior. Production calls ``begin``
    before workers start, which enables startup gating. A fresh start is
    trusted to be at a safe location, so the area is considered clear at
    ``begin``: startup buffs and then timer casts must complete before combat
    resumes. A scan that finds mobs downgrades the milestone, which reopens
    combat for the clear pass; recovered hunts re-arm the wait from scratch.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self._managed = False
        # Production supplies the milestones that have real workers behind
        # them. Defaults preserve the strict behavior of standalone fixtures.
        self._require_buffs = True
        self._require_timers = True
        # Combat may clear a populated area before the first successful scan
        # (a trusted safe start downgraded by the first scan, or a recovered
        # landing). Startup actions remain gated until discovery confirms the
        # area is empty, so this never releases normal post-clear combat
        # prematurely.
        self._allow_combat_before_area_clear = True
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

    def begin(
        self,
        *,
        require_buffs: bool = True,
        require_timers: bool = True,
    ) -> None:
        """Enable startup management for the configured worker milestones.

        A fresh hunt is trusted to start at a safe location (the user positions
        the character there before starting), so the area is considered clear
        immediately: startup buffs and timer casts are not held back by the
        first discovery scan. A scan that finds mobs downgrades the milestone,
        which reopens combat for the clear pass.
        """
        with self._lock:
            self._managed = True
            self._require_buffs = require_buffs
            self._require_timers = require_timers
            self._allow_combat_before_area_clear = True
            self._clear_milestones()
            # Trusted safe start: the area is empty by definition, so startup
            # actions may run immediately instead of waiting for the first scan.
            self.area_clear.set()

    def begin_new_hunt(self) -> None:
        """Advance to a new hunt and reset its startup milestones atomically."""
        with self._lock:
            self._generation += 1
            if self._managed:
                # A recovered hunt (sit/stand, teleport) must not assume the
                # landing area is empty — mobs may have moved in while the
                # character was away. Discovery/tracking and attack stay live
                # before the first clear scan; once the scan confirms clear,
                # buffs then timers gate the normal hunt again.
                self._allow_combat_before_area_clear = True
                self._clear_milestones()

    def mark_area_clear(
        self,
        clear: bool = True,
        *,
        expected_generation: int | None = None,
    ) -> bool:
        """Record discovery state only if it belongs to the active hunt."""
        with self._lock:
            if (
                expected_generation is not None
                and expected_generation != self._generation
            ):
                return False
            if clear:
                self.area_clear.set()
            else:
                self.area_clear.clear()
            return True

    def mark_buffs_done(self, *, expected_generation: int | None = None) -> bool:
        """Complete buffs only for the active hunt generation."""
        with self._lock:
            if (
                expected_generation is not None
                and expected_generation != self._generation
            ):
                return False
            self.buffs_done.set()
            return True

    def mark_timers_done(self, *, expected_generation: int | None = None) -> bool:
        """Complete timer startup only for the active hunt generation."""
        with self._lock:
            if (
                expected_generation is not None
                and expected_generation != self._generation
            ):
                return False
            self.timers_done.set()
            return True

    def is_combat_ready(self) -> bool:
        """Return whether startup milestones permit combat.

        Normal skill timers are part of hunt startup. They may be staggered,
        but every configured timer must fire once after buffs before the attack
        loop is released. The complete read is locked with generation resets
        so the attack loop cannot observe a mixed old/new hunt state.
        """
        with self._lock:
            # Initial starts trust the user's safe position: the area-clear
            # milestone is satisfied from ``begin``, so combat waits for the
            # startup buffs/timers. A scan finding mobs downgrades the
            # milestone, and combat then remains available so the hunt can
            # clear that populated area; the next clear scan re-gates on the
            # completed startup actions. Unmanaged fixtures stay ready.
            if not self._managed:
                return True
            if not self.area_clear.is_set():
                # The landing area may already contain mobs (populated start
                # discovered by the first scan, or a post-sit/post-teleport
                # landing). Let combat clear that area first; discovery then
                # closes this pre-clear window and startup actions run before
                # the normal hunt resumes.
                return self._allow_combat_before_area_clear
            return self.buffs_done.is_set() and self.timers_done.is_set()

    def _clear_milestones(self) -> None:
        # Called only while _lock is held. Keeping this operation together with
        # generation changes prevents a sit-to-hunt transition from publishing
        # a new generation with stale completion events.
        self.area_clear.clear()
        if self._require_buffs:
            self.buffs_done.clear()
        else:
            self.buffs_done.set()
        if self._require_timers:
            self.timers_done.clear()
        else:
            self.timers_done.set()


__all__ = ["HuntStartupSequence"]
