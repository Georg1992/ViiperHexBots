"""DangerDetector — isolated danger observer.

Runs in its own worker thread.  Polls ``PlayerVitals`` for HP and
reads ``CharacterState`` for surround / nearby mobs.

* Any HP drop → record damage (sit/heal consumers)
* Any HP drop → record recent damage
* Surrounded + recent damage, or HP below 50% + recent damage → urgent
  danger teleport (when allowed)
* Nearby mobs remain a healing threat even when they do not qualify for TP
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
    from pybot.runtime.teleport import TeleportController
    from pybot.runtime.character_state import CharacterState

CRITICAL_HP_RATIO = 0.5


class DangerLevel(IntEnum):
    """Current character threat level, ordered from safe to critical."""

    SAFE = 0
    DANGER = 1
    CRITICAL = 2


class DangerDetector:
    """Observes HP and CharacterState; urgent TP requires fresh damage."""

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
        """Ongoing loop: poll HP, check urgent danger, sleep."""
        while not self._ctx.is_stopped():
            self._poll_hp()
            self._check_urgent_danger()
            self._ctx.stop_event.wait(WORKER_POLL_INTERVAL_S)

    def _poll_hp(self) -> None:
        """Read HP from shared vitals and record every HP drop as damage."""
        with self._damage_lock:
            hp, _hp_max = self._vitals.hp_pair()
            if hp is None:
                self._prev_hp = None
                return

            if self._prev_hp is not None and hp < self._prev_hp:
                self._damage_detected = True
                self._last_damage_mono = time.monotonic()
            self._prev_hp = hp

    def _check_urgent_danger(self) -> None:
        """Ask the sit worker to escape any danger and restart the hunt.

        Sitting is the single hunt-break path. The sit worker owns the safe
        teleport, sit/stand toggle, SP recovery, and generation restart; this
        observer only raises the request and never teleports independently.
        """
        if not self._ctx.should_run_workers():
            return
        # A nearby mob by itself is ordinary combat, not a reason to break
        # the hunt. Danger-driven sitting is damage-based only; every recent
        # HP drop is handed to the sit worker, which decides how to reach a
        # safe place and recover SP. Critical level still has the same
        # damage+surround/low-HP semantics for callers that inspect it.
        if not self.has_recent_damage(HP_HEAL_DAMAGE_QUIET_S):
            return
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

    def reset_after_teleport(self) -> None:
        """Forget damage state after a successful teleport into a new area."""
        with self._damage_lock:
            hp, _hp_max = self._vitals.hp_pair()
            self._damage_detected = False
            self._last_damage_mono = None
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

    def pop_damage_detected(self) -> bool:
        """Return and clear the damage-detected flag for consumers (sit worker)."""
        with self._damage_lock:
            if self._damage_detected:
                self._damage_detected = False
                return True
            return False
