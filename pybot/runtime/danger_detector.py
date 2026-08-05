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
# While seated, HP must keep flowing so seated danger can be seen. The status
# panel OCR feed can go quiet (window inactive / panel closed) without any
# damage signal; log that blind spot and request an emergency escape instead
# of allowing a blind seated character to die.
HP_STALE_LOG_INTERVAL_MS = 5000
HP_STALE_THRESHOLD_S = 3.0


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
        self._last_hp_stale_log_ms = 0
        self._stale_escape_requested = False

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
                self._log_hp_stale_if_sitting()
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
            # The HP feed may publish the same value forever (window inactive,
            # panel closed, OCR failing). While seated this blinds danger
            # protection, so surface it in the log at a throttled cadence.
            self._log_hp_stale_if_sitting()

        if damage_seen:
            # Queue damage for an active seated recovery session only.
            # Critical damage also gets its own independent hunting escape
            # signal, so critical protection does not depend on sit being
            # enabled or configured.
            level = self.danger_level()
            # A danger-sit request belongs only to an already-owned seated
            # recovery session. While hunting, ordinary damage must not enter
            # the sit worker at all: the independent critical event is the
            # sole owner of a hunting escape. The previous unconditional
            # request made every small HP loss block combat and caused the sit
            # worker to run immediately after a critical teleport.
            sitting = getattr(self._ctx, "sitting_event", None)
            suspend = getattr(self._ctx, "discovery_suspend", None)
            escape = getattr(self._ctx, "danger_escape_active", None)
            is_sitting = False
            is_teleporting = False
            is_escaping = False
            is_set = getattr(sitting, "is_set", None)
            if callable(is_set):
                value = is_set()
                is_sitting = type(value) is bool and value
            suspend_is_set = getattr(suspend, "is_set", None)
            if callable(suspend_is_set):
                value = suspend_is_set()
                is_teleporting = type(value) is bool and value
            escape_is_set = getattr(escape, "is_set", None)
            if callable(escape_is_set):
                value = escape_is_set()
                is_escaping = type(value) is bool and value
            request = getattr(self._ctx, "request_danger_sit", None)
            # During an escape teleport, ``sitting_event`` is deliberately held
            # as an input gate. It does not mean the character entered SP
            # recovery. Never turn damage sampled during teleport settle into a
            # new sit session; the critical/urgent owner already owns escape.
            sit_queued = bool(
                request()
                if is_sitting
                and not is_teleporting
                and not is_escaping
                and callable(request)
                else False
            )
            critical_queued = False
            # Critical damage must remain queued even while a sit/teleport
            # gate is held. The critical worker temporarily holds
            # ``sitting_event`` during danger teleport; a new HP drop during
            # settle would otherwise create only ``danger_sit_requested``.
            # The sit worker intentionally leaves critical hunting damage
            # alone, so that request would then survive the escape forever and
            # keep ``should_run_combat()`` false. Seated recovery consumes the
            # mirrored critical request when it owns the damage event.
            if level is DangerLevel.CRITICAL:
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

    def _log_hp_stale_if_sitting(self) -> None:
        """Diagnose and fail safe when seated HP reads stop being observed.

        A frozen HP feed makes the detector look healthy while the character is
        taking damage it cannot see. The first stale episode therefore queues a
        danger-sit request; the sit owner performs the actual emergency escape.
        """
        sitting = getattr(self._ctx, "sitting_event", None)
        is_set = getattr(sitting, "is_set", None)
        is_sitting = False
        if callable(is_set):
            value = is_set()
            is_sitting = type(value) is bool and value
        if not is_sitting:
            self._last_hp_stale_log_ms = 0
            self._stale_escape_requested = False
            return
        sample = self._vitals.hp_sample()
        hp, _hp_max, hp_observed_ms, _hp_changed_ms = sample
        now_ms = int(time.monotonic() * 1000)
        stale_s = (now_ms - hp_observed_ms) / 1000.0 if hp_observed_ms > 0 else -1.0
        if hp_observed_ms > 0 and stale_s < HP_STALE_THRESHOLD_S:
            self._last_hp_stale_log_ms = 0
            self._stale_escape_requested = False
            return
        if now_ms - self._last_hp_stale_log_ms < HP_STALE_LOG_INTERVAL_MS:
            return
        self._last_hp_stale_log_ms = now_ms
        logger = getattr(self._ctx, "logger", None)
        behavior = getattr(logger, "behavior", None)
        if callable(behavior):
            behavior(
                f"[DANGER] HP feed stale while sitting hp={hp} "
                f"staleFor={stale_s:.0f}s — damage while seated may be invisible"
            )
        # A stale feed is itself an unsafe seated state. Request one escape
        # for this stale episode; the sit worker owns the actual teleport and
        # will retry if input/teleport is temporarily unavailable. Do not
        # enqueue repeatedly on every 50ms detector poll.
        if not self._stale_escape_requested:
            request = getattr(self._ctx, "request_danger_sit", None)
            if callable(request):
                request()
                self._stale_escape_requested = True
                if callable(behavior):
                    behavior(
                        "[DANGER] stale seated HP feed — requesting emergency escape"
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
