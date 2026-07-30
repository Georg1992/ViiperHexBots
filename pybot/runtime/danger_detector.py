"""Centralized danger detection — observes game state, triggers teleport.

The :class:`DangerDetector` is an **observer**: callers *feed* it data
(HP, track positions) and it decides when danger exists.  No module
calls a danger method directly — they just provide the latest readings.

Currently observed signals:

* ``feed_hp(hp)`` — HP drop (universal)
* ``feed_tracks(char_x, char_y, all_mobs)`` — surrounded (only for mobs
  with ``has_custom_behavior()`` — Anubis for now)

When any signal fires, :meth:`_fire` executes danger teleport via
``MobBehavior.execute_danger_teleport`` (wing-first key).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pybot.runtime.input.input_backend import InputBackend
    from pybot.runtime.mob_behaviors import MobBehavior


class DangerDetector:
    """Observes HP and track data; triggers danger teleport automatically."""

    def __init__(
        self,
        ctx,
        hunt_mode,
        input_backend: InputBackend,
        mob_behavior: MobBehavior,
    ) -> None:
        self._ctx = ctx
        self._hunt_mode = hunt_mode
        self._input = input_backend
        self._mob_behavior = mob_behavior
        self._prev_hp: int | None = None

    # ── Data-in methods ──────────────────────────────────────────

    def feed_hp(self, hp: int | None) -> bool:
        """Observe current HP.  Triggers on HP drop.

        Returns ``True`` when danger teleport was executed.
        """
        if hp is not None and self._prev_hp is not None and hp < self._prev_hp:
            if self._fire("hp_drop"):
                self._prev_hp = hp
                return True
            return False
        self._prev_hp = hp
        return False

    def feed_tracks(
        self,
        char_x: int,
        char_y: int,
        all_mobs: list[tuple[int, int]],
    ) -> bool:
        """Observe tracked mob positions.  Triggers on surrounded.

        Surrounded is only active for mobs with custom behavior
        (Anubis for now).  Returns ``True`` when danger teleport
        was executed.
        """
        if not self._mob_behavior.has_custom_behavior():
            return False
        is_surrounded, reason = self._mob_behavior.is_surrounded(
            char_x, char_y, all_mobs,
        )
        if is_surrounded:
            return self._fire(f"surrounded {reason}")
        return False

    # ── Internal ─────────────────────────────────────────────────

    def _fire(self, reason: str) -> bool:
        """Execute danger teleport unconditionally."""
        return self._mob_behavior.execute_danger_teleport(
            self._ctx, self._hunt_mode, self._input, reason=reason,
        )
