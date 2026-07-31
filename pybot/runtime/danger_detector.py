"""DangerDetector — isolated danger observer.

Runs in its own worker thread.  Polls ``PlayerVitals`` for HP and
reads ``CharacterState`` for surround / nearby mobs.

* HP drop → record damage + danger teleport (when allowed)
* Surrounded → danger teleport (when allowed)
* Heal-skill uses ``has_nearby_threat`` / ``has_recent_damage`` for a
  deterministic safe-to-heal check
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from pybot.runtime.constants import WORKER_POLL_INTERVAL_S

if TYPE_CHECKING:
    from pybot.game_state import PlayerVitals
    from pybot.runtime.teleport import TeleportController
    from pybot.runtime.character_state import CharacterState

CRITICAL_HP_RATIO = 0.5


class DangerDetector:
    """Observes HP and CharacterState; triggers danger teleport automatically."""

    def __init__(
        self,
        ctx,
        teleport: TeleportController,
        character_state: CharacterState,
        vitals: PlayerVitals,
    ) -> None:
        self._ctx = ctx
        self._teleport = teleport
        self._character_state = character_state
        self._vitals = vitals
        self._prev_hp: int | None = None
        self._damage_lock = threading.Lock()
        self._damage_detected: bool = False
        self._last_damage_mono: float | None = None

    def run(self) -> None:
        """Ongoing loop: poll HP + surround state, detect danger, sleep."""
        while not self._ctx.is_stopped():
            self._poll_hp()
            if self._ctx.should_allow_danger_teleport():
                self._poll_surround()
            self._ctx.stop_event.wait(WORKER_POLL_INTERVAL_S)

    def _poll_hp(self) -> None:
        """Read HP from shared vitals; record drops; teleport when allowed."""
        hp, hp_max = self._vitals.hp_pair()
        if hp is None:
            self._prev_hp = None
            return

        if self._prev_hp is not None and hp < self._prev_hp:
            is_critical = (
                hp_max is not None
                and hp_max > 0
                and hp / hp_max <= CRITICAL_HP_RATIO
            )
            with self._damage_lock:
                self._damage_detected = True
                self._last_damage_mono = time.monotonic()
            if self._ctx.should_allow_danger_teleport():
                reason = "critical_hp" if is_critical else "hp_drop"
                self._teleport.danger_teleport(reason=reason)
        self._prev_hp = hp

    def _poll_surround(self) -> None:
        """Teleport when CharacterState reports surround."""
        if self._character_state.is_surrounded:
            reason = self._character_state.surrounded_reason
            self._teleport.danger_teleport(reason=f"surrounded {reason}")

    def has_nearby_threat(self) -> bool:
        """True when any mob is near the character (tracks or visual)."""
        state = self._character_state
        return (
            bool(state.is_surrounded)
            or int(state.nearby_mob_count) > 0
            or int(state.nearby_any_mobs_count) > 0
        )

    def has_recent_damage(self, within_s: float) -> bool:
        """True if an HP drop was observed within the last ``within_s`` seconds."""
        with self._damage_lock:
            if self._last_damage_mono is None:
                return False
            return (time.monotonic() - self._last_damage_mono) < within_s

    def pop_damage_detected(self) -> bool:
        """Return and clear the damage-detected flag for consumers (sit worker)."""
        with self._damage_lock:
            if self._damage_detected:
                self._damage_detected = False
                return True
            return False
