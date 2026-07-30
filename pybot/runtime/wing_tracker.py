"""WingTracker — fly-wing count and restock state.

Owns the wingcount counter and fly_wings_exhausted flag, keeping
this concern out of HuntRuntimeContext.
"""

from __future__ import annotations

import threading


class WingTracker:
    """Tracks remaining fly wings and restock state during a hunt session.

    Thread-safe: all mutations use ``_lock``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # AHK ``wingcount``: remaining fly wings; restocked by GetFlyWings.
        self._wingcount: int = 0
        # Set when storage has no wings left — stop GetFlyWings for this hunt.
        self._fly_wings_exhausted: bool = False

    # ── Read-only properties (locked) ────────────────────────────

    @property
    def wingcount(self) -> int:
        with self._lock:
            return self._wingcount

    @wingcount.setter
    def wingcount(self, value: int) -> None:
        with self._lock:
            self._wingcount = value

    @property
    def fly_wings_exhausted(self) -> bool:
        with self._lock:
            return self._fly_wings_exhausted

    @fly_wings_exhausted.setter
    def fly_wings_exhausted(self, value: bool) -> None:
        with self._lock:
            self._fly_wings_exhausted = value

    # ── Operations ───────────────────────────────────────────────

    def note_teleport(self, *, open_storage_steps: bool, take_fly_wings: bool) -> None:
        """AHK Teleport: decrement wing counter when Take Fly Wings is on.

        Called after every teleport key press. Thread-safe.
        """
        with self._lock:
            if (
                open_storage_steps
                and take_fly_wings
                and not self._fly_wings_exhausted
                and self._wingcount > 0
            ):
                self._wingcount -= 1

    def should_restock(
        self, *, open_storage_steps: bool, take_fly_wings: bool, fly_wings_amount: int
    ) -> bool:
        """True when GetFlyWings should run (enabled, amount set, count 0)."""
        return (
            bool(open_storage_steps)
            and take_fly_wings
            and fly_wings_amount > 0
            and not self._fly_wings_exhausted
            and self._wingcount <= 0
        )

    def mark_exhausted(self) -> None:
        """Stop fly-wing restock for this hunt."""
        with self._lock:
            self._fly_wings_exhausted = True
            self._wingcount = 0

    def set_count(self, count: int) -> None:
        """Set the wing count after a successful restock."""
        with self._lock:
            self._wingcount = count

    def reset(self) -> None:
        """Reset to initial state for a new hunt session."""
        with self._lock:
            self._wingcount = 0
            self._fly_wings_exhausted = False
