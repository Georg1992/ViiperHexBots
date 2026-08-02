"""DangerDetector — isolated HP damage observer.

Runs in its own worker thread. Any observed HP drop records damage and queues
one danger-sit request. Danger decisions rely only on received HP damage.
"""

from __future__ import annotations

import threading
import time
from enum import IntEnum

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import (
    HP_HEAL_DAMAGE_QUIET_S,
    WORKER_POLL_INTERVAL_S,
)

CRITICAL_HP_RATIO = 0.5
CRITICAL_DAMAGE_RATIO = 0.2


class DangerLevel(IntEnum):
    """Current damage level, ordered from safe to critical."""

    SAFE = 0
    DANGER = 1
    CRITICAL = 2


class DangerDetector:
    """Observes HP damage; the sit worker owns danger escape."""

    def __init__(self, ctx, vitals: PlayerVitals) -> None:
        self._ctx = ctx
        self._vitals = vitals
        self._prev_hp: int | None = None
        self._damage_lock = threading.Lock()
        self._last_damage_mono: float | None = None
        self._last_damage_ratio: float | None = None

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
                # A transient unreadable HP sample must not erase the last
                # known baseline. Otherwise the next valid lower sample is
                # treated as a new baseline and damage while sitting is lost.
                return

            if self._prev_hp is not None and hp < self._prev_hp:
                previous_hp = self._prev_hp
                self._last_damage_mono = time.monotonic()
                self._last_damage_ratio = (
                    (previous_hp - hp) / previous_hp
                    if previous_hp > 0
                    else None
                )
                damage_seen = True
            self._prev_hp = hp

        if damage_seen:
            # Queue every damage event for an active seated recovery session.
            # Critical damage also gets its own independent hunting escape
            # signal, so critical protection does not depend on sit being
            # enabled or configured.
            request = getattr(self._ctx, "request_danger_sit", None)
            if callable(request):
                request()
            sitting = getattr(self._ctx, "sitting_event", None)
            is_sitting = bool(
                sitting is not None
                and callable(getattr(sitting, "is_set", None))
                and sitting.is_set()
            )
            if not is_sitting and self.danger_level() is DangerLevel.CRITICAL:
                request_critical = getattr(self._ctx, "request_critical_danger", None)
                if callable(request_critical):
                    request_critical()

    def danger_level(self) -> DangerLevel:
        """Return SAFE, DANGER, or CRITICAL using received damage only.

        Critical danger requires recent damage plus either HP below 50% or a
        per-tick HP loss greater than 20% of the previous HP sample.
        """
        hp, hp_max = self._vitals.hp_pair()
        damaged = self.has_recent_damage(HP_HEAL_DAMAGE_QUIET_S)
        with self._damage_lock:
            critical_damage = (
                self._last_damage_ratio is not None
                and self._last_damage_ratio > CRITICAL_DAMAGE_RATIO
            )
        critical_hp = (
            hp is not None
            and hp_max is not None
            and hp_max > 0
            and hp / hp_max < CRITICAL_HP_RATIO
        )
        if damaged and (critical_damage or critical_hp):
            return DangerLevel.CRITICAL
        if damaged:
            return DangerLevel.DANGER
        return DangerLevel.SAFE

    def reset_after_teleport(self, tp_start_mono: float | None = None) -> None:
        """Forget pre-teleport damage without consuming sit requests.

        The current HP becomes the new baseline so the first sample after a
        teleport is not mistaken for damage merely because the area changed.
        Damage observed during the settle window is preserved.
        """
        with self._damage_lock:
            if (
                tp_start_mono is not None
                and self._last_damage_mono is not None
                and self._last_damage_mono >= tp_start_mono
            ):
                return
            hp, _hp_max = self._vitals.hp_pair()
            self._prev_hp = hp
            self._last_damage_mono = None
            self._last_damage_ratio = None


    def is_safe_for_heal(self) -> bool:
        """True when a self-heal may run without recent damage."""
        if self._ctx.discovery_suspend.is_set():
            return False
        return self.danger_level() is DangerLevel.SAFE

    def has_recent_damage(self, within_s: float) -> bool:
        """True if an HP drop was observed within the last ``within_s`` seconds."""
        with self._damage_lock:
            if self._last_damage_mono is None:
                return False
            return (time.monotonic() - self._last_damage_mono) < within_s
