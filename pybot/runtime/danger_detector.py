"""DangerDetector — isolated danger observer.

Runs in its own worker thread.  Polls ``PlayerVitals`` for HP and
reads ``CharacterState`` for surround / nearby mobs.

* Any HP drop → record damage (sit/heal consumers)
* Critical HP (≤50%) → urgent danger teleport (when allowed)
* Surrounded / single nearby mob → never urgent-teleport; surround is
  informational for heal threat only (needs ≥2 opposite-side tracks)
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from pybot.runtime.constants import (
    HP_HEAL_DAMAGE_QUIET_S,
    WORKER_POLL_INTERVAL_S,
)

if TYPE_CHECKING:
    from pybot.game_state import PlayerVitals
    from pybot.runtime.teleport import TeleportController
    from pybot.runtime.character_state import CharacterState

CRITICAL_HP_RATIO = 0.5


class DangerDetector:
    """Observes HP and CharacterState; urgent TP only on critical HP."""

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
        """Ongoing loop: poll HP, detect critical danger, sleep."""
        while not self._ctx.is_stopped():
            self._poll_hp()
            self._ctx.stop_event.wait(WORKER_POLL_INTERVAL_S)

    def _poll_hp(self) -> None:
        """Read HP from shared vitals; record drops; TP only when critical."""
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
            # Urgent teleport only on critical danger — not every chip hit,
            # and not from surround / a single nearby tracked mob.
            if is_critical and self._ctx.should_allow_danger_teleport():
                self._teleport.danger_teleport(reason="critical_hp")
        self._prev_hp = hp

    def has_nearby_threat(self) -> bool:
        """True when any mob is near the character (tracks or visual)."""
        state = self._character_state
        return (
            bool(state.is_surrounded)
            or int(state.nearby_mob_count) > 0
            or int(state.nearby_any_mobs_count) > 0
        )

    def is_safe_for_heal(self) -> bool:
        """True when a self-heal may run without an active nearby threat.

        The post-teleport grace window is intentionally safe; otherwise the
        character must have no nearby visual/tracked threat and no recent HP
        damage. Discovery suspension always wins over healing.
        """
        if self._ctx.discovery_suspend.is_set():
            return False
        if self._ctx.in_post_teleport_heal_window():
            return True
        return not self.has_nearby_threat() and not self.has_recent_damage(
            HP_HEAL_DAMAGE_QUIET_S
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
