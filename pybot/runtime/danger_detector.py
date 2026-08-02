"""DangerDetector — isolated danger observer.

Runs in its own worker thread.  Polls ``PlayerVitals`` for HP and
reads ``CharacterState`` for surround / nearby mobs.

* Any HP drop → record recent damage and queue one danger-sit request
* Nearby mobs remain a healing threat for healing decisions; the sit worker
  owns the escape sequence
"""

from __future__ import annotations

import threading
import time
from enum import IntEnum
from typing import TYPE_CHECKING

from pybot.runtime.constants import (
    HP_HEAL_DAMAGE_QUIET_S,
    WORKER_POLL_INTERVAL_S,
)

if TYPE_CHECKING:
    from pybot.game_state import PlayerVitals
    from pybot.runtime.character_state import CharacterState

CRITICAL_HP_RATIO = 0.5


class DangerLevel(IntEnum):
    """Current character threat level, ordered from safe to critical."""

    SAFE = 0
    DANGER = 1
    CRITICAL = 2


class DangerDetector:
    """Observes HP and CharacterState; the sit worker owns danger escape."""

    def __init__(
        self,
        ctx,
        character_state: CharacterState,
        vitals: PlayerVitals,
    ) -> None:
        self._ctx = ctx
        self._character_state = character_state
        self._vitals = vitals
        self._prev_hp: int | None = None
        self._damage_lock = threading.Lock()
        self._last_damage_mono: float | None = None

    def run(self) -> None:
        """Ongoing loop: poll HP and sleep until stopped."""
        while not self._ctx.is_stopped():
            self._poll_hp()
            self._ctx.stop_event.wait(WORKER_POLL_INTERVAL_S)

    def _poll_hp(self) -> None:
        """Read HP and queue one danger request for each observed drop."""
        damage_seen = False
        with self._damage_lock:
            hp, _hp_max = self._vitals.hp_pair()
            if hp is None:
                self._prev_hp = None
                return

            if self._prev_hp is not None and hp < self._prev_hp:
                self._last_damage_mono = time.monotonic()
                damage_seen = True
            self._prev_hp = hp

        if damage_seen:
            # Raise the sole sit/danger signal after releasing the damage lock.
            # The sit worker can then safely contend for the session boundary.
            request = getattr(self._ctx, "request_danger_sit", None)
            if callable(request):
                request()

    def danger_level(self) -> DangerLevel:
        """Return SAFE, DANGER, or CRITICAL from the current shared state.

        Recent damage is required for CRITICAL. Surround + damage and HP below
        50% + damage are both critical. Any remaining recent damage or visible
        nearby threat is DANGER; only neither condition is SAFE.
        """
        hp, hp_max = self._vitals.hp_pair()
        damaged = self.has_recent_damage(HP_HEAL_DAMAGE_QUIET_S)
        surrounded = bool(self._character_state.is_surrounded)
        critical_hp = (
            hp is not None
            and hp_max is not None
            and hp_max > 0
            and hp / hp_max < CRITICAL_HP_RATIO
        )
        if damaged and (surrounded or critical_hp):
            return DangerLevel.CRITICAL
        if damaged or self.has_nearby_threat():
            return DangerLevel.DANGER
        return DangerLevel.SAFE

    def reset_after_teleport(self, tp_start_mono: float | None = None) -> None:
        """Forget pre-teleport damage without consuming sit requests.

        When called with the teleport start time, damage recorded during the
        settle window is retained. With no timestamp, clear the current sample
        for direct/manual area resets.
        """
        with self._damage_lock:
            if (
                tp_start_mono is not None
                and self._last_damage_mono is not None
                and self._last_damage_mono >= tp_start_mono
            ):
                return
            self._last_damage_mono = None

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

        The post-teleport grace window only permits healing when the landing
        area is still clear. An active nearby threat or recent damage always
        blocks healing, because urgent teleport has priority.
        """
        if self._ctx.discovery_suspend.is_set():
            return False
        return self.danger_level() is DangerLevel.SAFE

    def has_recent_damage(self, within_s: float) -> bool:
        """True if an HP drop was observed within the last ``within_s`` seconds."""
        with self._damage_lock:
            if self._last_damage_mono is None:
                return False
            return (time.monotonic() - self._last_damage_mono) < within_s
