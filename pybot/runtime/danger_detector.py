"""DangerDetector — isolated danger observer.

Callers feed data in; the detector decides danger and teleports.
Nothing comes back out.  No other module knows danger exists.

Inputs:
* ``feed_hp(hp, hp_max=None)`` — HP drop triggers teleport;
  drops below 50% of max are logged as critical
* ``feed_tracks(char_x, char_y, all_mobs)`` — surrounded triggers
  teleport (only for mobs with custom behavior — Anubis for now)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pybot.runtime.input.input_backend import InputBackend
    from pybot.runtime.mob_behaviors import MobBehavior

CRITICAL_HP_RATIO = 0.5


class DangerDetector:
    """Observes HP and track data; triggers danger teleport automatically."""

    def __init__(
        self,
        ctx,
        input_backend: InputBackend,
        mob_behavior: MobBehavior,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._mob_behavior = mob_behavior
        self._prev_hp: int | None = None

    # ── Data-in methods ──────────────────────────────────────────

    def feed_hp(self, hp: int | None, hp_max: int | None = None) -> bool:
        """Observe current HP.  Teleports on any HP drop.

        Drops below 50% of max are logged as ``critical_hp`` — the
        action is the same (teleport).  Callers that know max HP pass
        it along for richer logging.

        Returns ``True`` when the detector teleported.
        """
        if hp is not None and self._prev_hp is not None and hp < self._prev_hp:
            # Check if this is a critical drop (below 50% of max)
            is_critical = (
                hp_max is not None
                and hp_max > 0
                and hp / hp_max < CRITICAL_HP_RATIO
            )
            if is_critical:
                self._mob_behavior.execute_danger_teleport(
                    self._ctx, self._input, reason="critical_hp",
                )
            else:
                self._mob_behavior.execute_danger_teleport(
                    self._ctx, self._input, reason="hp_drop",
                )
            self._prev_hp = hp
            return True
        self._prev_hp = hp
        return False

    def feed_tracks(
        self,
        char_x: int,
        char_y: int,
        all_mobs: list[tuple[int, int]],
    ) -> bool:
        """Observe tracked mob positions.  Teleports if surrounded.

        Only active for mobs with custom behavior (Anubis for now).
        Returns ``True`` when the detector teleported.
        """
        if not self._mob_behavior.has_custom_behavior():
            return False
        is_surrounded, reason = self._mob_behavior.is_surrounded(
            char_x, char_y, all_mobs,
        )
        if is_surrounded:
            self._mob_behavior.execute_danger_teleport(
                self._ctx, self._input, reason=f"surrounded {reason}",
            )
            return True
        return False
