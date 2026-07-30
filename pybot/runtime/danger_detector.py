"""DangerDetector — isolated danger observer.

Runs in its own worker thread.  Polls ``PlayerVitals`` for HP, and is
called from the attack loop with tracked mob positions for surround
detection.  Nothing comes back out.  No other module knows danger exists.

Inputs:
* ``run()`` — polls ``hp_pair()`` from PlayerVitals; HP drops trigger
  teleport; drops below 50% of max are logged as critical
* ``feed_tracks(char_x, char_y, all_mobs)`` — surrounded triggers
  teleport (only for mobs with custom behavior — Anubis for now)

Teleport execution delegated to :class:`TeleportController`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pybot.runtime.constants import WORKER_POLL_INTERVAL_S

if TYPE_CHECKING:
    from pybot.game_state import PlayerVitals
    from pybot.runtime.teleport import TeleportController

CRITICAL_HP_RATIO = 0.5


class DangerDetector:
    """Observes HP and track data; triggers danger teleport automatically."""

    def __init__(
        self,
        ctx,
        teleport: TeleportController,
        mob_behavior,
        vitals: PlayerVitals,
    ) -> None:
        self._ctx = ctx
        self._teleport = teleport
        self._mob_behavior = mob_behavior
        self._vitals = vitals
        self._prev_hp: int | None = None
        self._damage_detected: bool = False

    # ── Worker loop ───────────────────────────────────────────────

    def run(self) -> None:
        """Ongoing loop: poll vitals HP, detect drops, sleep."""
        while not self._ctx.is_stopped():
            self._poll_hp()
            self._ctx.stop_event.wait(WORKER_POLL_INTERVAL_S)

    def _poll_hp(self) -> None:
        """Read HP from shared vitals and teleport if dropped."""
        hp, hp_max = self._vitals.hp_pair()
        if hp is None:
            self._prev_hp = None
            return

        if self._prev_hp is not None and hp < self._prev_hp:
            is_critical = (
                hp_max is not None
                and hp_max > 0
                and hp / hp_max < CRITICAL_HP_RATIO
            )
            self._damage_detected = True
            if is_critical:
                self._teleport.danger_teleport(reason="critical_hp")
            else:
                self._teleport.danger_teleport(reason="hp_drop")
        self._prev_hp = hp

    def pop_damage_detected(self) -> bool:
        """Return and clear the damage-detected flag for consumers (sit worker)."""
        if self._damage_detected:
            self._damage_detected = False
            return True
        return False

    # ── Track data ────────────────────────────────────────────────

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
            self._teleport.danger_teleport(reason=f"surrounded {reason}")
            return True
        return False
