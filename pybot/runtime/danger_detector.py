"""Independent HP damage observation and urgent-signal publication.

This thread never performs gameplay input. It stores damage state, publishes
simple requests, and signals cancellation of an in-flight input macro when
critical damage is observed. GameplayLoop owns the resulting teleport.
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
from pybot.runtime.event_utils import event_is_set

CRITICAL_HP_RATIO = 0.5
CRITICAL_DAMAGE_RATIO = 0.2

class DangerLevel(IntEnum):
    """Current damage level, ordered from safe to critical."""

    SAFE = 0
    DANGER = 1
    CRITICAL = 2


class DangerDetector:
    """Observes HP damage and publishes state/signals; it never performs input."""

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
                # An unreadable sample is not a damage event and must not alter
                # the last valid baseline.
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
            # Wake/cancel the current gameplay input immediately. The gameplay
            # thread remains the sole owner of the eventual teleport; this
            # signal only makes a long input macro unwind promptly.
            level = self.danger_level()
            if level is DangerLevel.CRITICAL:
                cancel = getattr(self._ctx, "cancel_gameplay_input", None)
                if callable(cancel):
                    cancel()
            # Queue damage for an active seated recovery session only. Critical
            # damage gets a separate urgent signal; the gameplay loop owns the
            # resulting escape, so this observer never competes for input.
            # A danger-sit request belongs only to an already-owned seated
            # recovery session. While hunting, ordinary damage must not enter
            # the sit worker at all: the urgent event is consumed by
            # GameplayLoop. The previous unconditional
            # request made every small HP loss block combat and caused the sit
            # worker to run immediately after a critical teleport.
            sitting = getattr(self._ctx, "sitting_event", None)
            suspend = getattr(self._ctx, "discovery_suspend", None)
            escape = getattr(self._ctx, "danger_escape_active", None)
            critical_escape = getattr(
                self._ctx, "critical_danger_escape_active", None
            )
            # Lightweight/mock contexts report ``None`` here and are treated
            # as "no gate held" — the same as the previous defensive reads.
            is_sitting = event_is_set(sitting) is True
            is_teleporting = event_is_set(suspend) is True
            is_escaping = event_is_set(escape) is True
            is_critical_escape = event_is_set(critical_escape) is True
            # ``sitting_event`` is also borrowed by the critical hunting escape
            # as an input gate. Only a non-critical holder is true SP recovery.
            sit_owned = is_sitting and not is_critical_escape
            request = getattr(self._ctx, "request_danger_sit", None)
            # During any escape teleport, never enqueue a fresh sit session —
            # the escape owner already holds the character. Sit recovery still
            # observes settle damage via danger_level() after its own escape.
            sit_queued = bool(
                request()
                if sit_owned
                and not is_teleporting
                and not is_escaping
                and callable(request)
                else False
            )
            critical_queued = False
            # SP-sit recovery (including its own urgent escape) owns seated
            # danger on the gameplay thread. Queuing critical while sit_owned
            # would let another path clear sitting_event mid-regen: sit
            # abandons, SP is often already >5% so process_pending will not
            # restart recovery, and the hunt resumes incomplete / stale.
            # Hunting critical escapes are consumed by GameplayLoop; settle
            # damage during an escape may re-queue the signal.
            if level is DangerLevel.CRITICAL and not sit_owned:
                request_critical = getattr(self._ctx, "request_critical_danger", None)
                if callable(request_critical):
                    request_critical()
                    critical_queued = True
            logger = getattr(self._ctx, "logger", None)
            behavior = getattr(logger, "behavior", None)
            if callable(behavior):
                behavior(
                    f"[DANGER] HP drop previous={previous_hp} current={hp} "
                    f"loss={self._last_damage_ratio:.1%} level={level.name} "
                    f"sitQueued={sit_queued} criticalQueued={critical_queued}"
                )

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

        Only a *fresh* post-teleport HP observation may become the new
        baseline. The status-panel/OCR feed is blind through the loading
        transition, so the value at this instant may be the pre-teleport one:
        rebaselining from it makes the first fresh landing reading look like
        a phantom HP drop and re-triggers the escape — exactly the re-escape
        loop seen when seated recovery teleports. The baseline is held
        unknown until a fresh observation arrives. Damage observed during
        the settle window is preserved.
        """
        with self._damage_lock:
            if (
                tp_start_mono is not None
                and self._last_damage_mono is not None
                and self._last_damage_mono >= tp_start_mono
            ):
                return
            hp, _hp_max, hp_observed_ms, _hp_changed_ms = self._vitals.hp_sample()
            fresh = (
                True
                if tp_start_mono is None
                else hp_observed_ms >= int(tp_start_mono * 1000)
            )
            self._prev_hp = hp if fresh else None
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
