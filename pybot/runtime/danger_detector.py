"""Danger observation and the single owner of danger escapes.

The observer never queues gameplay requests and never owns input in production.
It only records the latest HP damage facts. ``DangerController`` is called by
``GameplayLoop`` (and by the synchronous sit recovery while it owns that
sequence) and is the only component allowed to turn those facts into a
teleport. One active transition marker covers the complete escape transaction;
there are no critical-vs-sit request handoffs.
"""

from __future__ import annotations

import inspect
import threading
import time
from enum import IntEnum
from typing import TYPE_CHECKING

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import (
    CRITICAL_PREEMPT_RELEASE_TIMEOUT_S,
    HP_HEAL_DAMAGE_QUIET_S,
    WORKER_POLL_INTERVAL_S,
)

if TYPE_CHECKING:
    from pybot.runtime.teleport import TeleportController

CRITICAL_HP_RATIO = 0.5
CRITICAL_DAMAGE_RATIO = 0.2


class DangerLevel(IntEnum):
    """Current damage level, ordered from safe to critical."""

    SAFE = 0
    DANGER = 1
    CRITICAL = 2


class DangerDetector:
    """Thread-safe HP damage facts; production code performs no input here."""

    def __init__(
        self,
        ctx=None,
        vitals: PlayerVitals | None = None,
        *,
        stop_event=None,
        wake_event=None,
    ) -> None:
        # ``ctx`` remains an optional compatibility argument for old fixtures.
        # Production supplies explicit lifecycle signals, so the observer has
        # no dependency on danger request gates or gameplay ownership.
        if vitals is None and isinstance(ctx, PlayerVitals):
            vitals, ctx = ctx, None
        self._ctx = ctx
        self._stop_event = stop_event
        self._wake_event = wake_event
        self._vitals = PlayerVitals() if vitals is None else vitals
        self._prev_hp: int | None = None
        self._damage_lock = threading.Lock()
        self._last_damage_mono: float | None = None
        self._last_damage_ratio: float | None = None
        # A real HP drop observed while already below the critical threshold
        # remains an urgent fact until HP recovers or the area is reset. The
        # short recent-damage window is still used for ordinary DANGER/heal
        # gating, but it must not make a low-HP damage event disappear before
        # gameplay gets a chance to press the escape key.
        self._critical_damage_seen = False
        self._damage_sequence = 0

    @property
    def vitals(self) -> PlayerVitals:
        return self._vitals

    @property
    def damage_sequence(self) -> int:
        """Monotonic count of observed HP drops."""
        with self._damage_lock:
            return self._damage_sequence

    @property
    def last_damage_mono(self) -> float | None:
        with self._damage_lock:
            return self._last_damage_mono

    def run(self) -> None:
        """Observe HP continuously; no event or input side effects in production."""
        while not self._is_stopped():
            self._poll_hp()
            self._stop_wait(WORKER_POLL_INTERVAL_S)

    def _is_stopped(self) -> bool:
        event = self._stop_event
        if event is not None:
            return bool(event.is_set())
        if self._ctx is None:
            return False
        return bool(self._ctx.is_stopped())

    def _stop_wait(self, timeout_s: float) -> None:
        event = self._stop_event
        if event is not None:
            event.wait(timeout_s)
            return
        if self._ctx is None:
            time.sleep(timeout_s)
            return
        self._ctx.stop_event.wait(timeout_s)

    def _poll_hp(self) -> None:
        """Publish only a new HP-damage fact."""
        with self._damage_lock:
            hp, _hp_max = self._vitals.hp_pair()
            if hp is None:
                return
            if self._prev_hp is None:
                self._prev_hp = hp
                return
            if hp >= self._prev_hp:
                self._prev_hp = hp
                # A known sample back above the critical threshold is explicit
                # recovery evidence. Unknown samples never reach this branch,
                # so they cannot accidentally clear an urgent latch.
                if not self._is_critical_hp(hp, _hp_max):
                    self._critical_damage_seen = False
                return
            previous_hp = self._prev_hp
            now = time.monotonic()
            self._prev_hp = hp
            self._last_damage_mono = now
            self._last_damage_ratio = (
                (previous_hp - hp) / previous_hp if previous_hp > 0 else None
            )
            self._critical_damage_seen = self._is_critical_hp(hp, _hp_max)
            self._damage_sequence += 1

        # The observer publishes facts and wakes the gameplay owner only. It
        # never requests a sit, sets a danger gate, cancels input, or presses a
        # key. The owner re-reads ``danger_level`` before acting.
        if self._wake_event is not None:
            self._wake_event.set()

    def danger_level(self) -> DangerLevel:
        """Classify the current damage fact without changing any runtime state."""
        hp, hp_max = self._vitals.hp_pair()
        damaged = self.has_recent_damage(HP_HEAL_DAMAGE_QUIET_S)
        with self._damage_lock:
            critical_damage = (
                self._last_damage_ratio is not None
                and self._last_damage_ratio > CRITICAL_DAMAGE_RATIO
            )
            critical_damage_seen = self._critical_damage_seen
        critical_hp = self._is_critical_hp(hp, hp_max)
        # A large hit is immediately critical while recent. A smaller hit that
        # leaves the player below 50% latches the same urgent state, so a slow
        # OCR/status update or a busy gameplay step cannot downgrade it to SAFE.
        # An unreadable sample is not recovery evidence; preserve the urgent
        # fact until a known HP value proves the player is back above the line.
        if critical_damage_seen:
            if hp is None or hp_max is None or hp_max <= 0:
                return DangerLevel.CRITICAL
            if critical_hp:
                return DangerLevel.CRITICAL
            with self._damage_lock:
                self._critical_damage_seen = False
        if damaged and (critical_damage or critical_hp):
            return DangerLevel.CRITICAL
        if damaged:
            return DangerLevel.DANGER
        return DangerLevel.SAFE

    @staticmethod
    def _is_critical_hp(
        hp: int | None,
        hp_max: int | None,
    ) -> bool:
        return (
            hp is not None
            and hp_max is not None
            and hp_max > 0
            and hp / hp_max < CRITICAL_HP_RATIO
        )

    def reset_after_teleport(self, tp_start_mono: float | None = None) -> None:
        """Start a new HP baseline after the teleport transaction.

        Damage observed after the teleport key was accepted belongs to the new
        area/landing transaction and must remain visible to the gameplay owner.
        Clearing it here can make a real landing hit disappear exactly when the
        sit/danger recovery is handing control back to the hunt loop.
        """
        with self._damage_lock:
            # A post-teleport observation may have arrived during settle. Keep
            # it as the baseline only when its observation clock is newer than
            # the teleport start; otherwise wait for the next real sample.
            _hp, _max_hp, observed_ms, _changed_ms = self._vitals.hp_sample()
            fresh = (
                tp_start_mono is None
                or observed_ms >= int(tp_start_mono * 1000)
            )
            damage_after_start = (
                tp_start_mono is not None
                and self._last_damage_mono is not None
                and self._last_damage_mono >= tp_start_mono
            )
            # A detector observation after the key boundary is itself proof
            # that the current HP belongs to the landing transaction, even if
            # the vitals observation clock was coarse or was published by a
            # lightweight adapter without the same timestamp precision.
            self._prev_hp = _hp if (fresh or damage_after_start) else None
            # Preserve a genuine landing hit. It belongs to the new area and
            # must remain urgent even after the ordinary recent-damage window.
            if damage_after_start:
                # Preserve the existing latch even when the landing sample is
                # unreadable; None/None is not proof that the danger ended.
                pass
            else:
                self._critical_damage_seen = False
            if not damage_after_start:
                self._last_damage_mono = None
                self._last_damage_ratio = None

    def is_safe_for_heal(self) -> bool:
        """True when no recent damage is active."""
        return self.danger_level() is DangerLevel.SAFE

    def has_recent_damage(self, within_s: float) -> bool:
        with self._damage_lock:
            if self._last_damage_mono is None:
                return False
            return time.monotonic() - self._last_damage_mono < within_s


class DangerController:
    """Single gameplay owner for danger transitions.

    ``process`` is called by ``GameplayLoop``. Sit recovery calls ``escape``
    directly while it owns its synchronous recovery sequence. Both routes use
    the same one active transition marker and the same teleport transaction.
    """

    def __init__(
        self,
        ctx,
        detector: DangerDetector,
        teleport: TeleportController,
        input_backend=None,
    ) -> None:
        self._ctx = ctx
        self._detector = detector
        self._teleport = teleport
        self._input_backend = input_backend
        self._lock = threading.RLock()

    @property
    def detector(self) -> DangerDetector:
        return self._detector

    def should_escape(self, *, seated: bool = False) -> bool:
        level = self._detector.danger_level()
        return level is DangerLevel.CRITICAL or (
            seated and level is DangerLevel.DANGER
        )

    def is_active(self) -> bool:
        return bool(self._ctx.danger_escape_active.is_set())

    def process(self, *, seated: bool = False) -> bool:
        """Process one danger tick; return True when a transition completed."""
        if self._ctx.is_stopped() or self._ctx.pause_event.is_set():
            return False
        if not self.should_escape(seated=seated):
            return False
        return self.escape(
            seated=seated,
            reason="sit_danger" if seated else "critical_hunt",
        )

    def _begin_input_for_escape(self) -> bool:
        """Wait briefly for a canceled input macro to release its lock.

        Critical danger must not disappear because storage/heal was still
        unwinding during the first 250 ms admission attempt. The cancellation
        above is asynchronous, so retry the bounded input-session admission
        until the emergency timeout expires, then leave the danger fact active
        for the next gameplay tick.
        """
        begin = getattr(self._input_backend, "begin_session", None)
        if not callable(begin):
            return True
        deadline = time.monotonic() + CRITICAL_PREEMPT_RELEASE_TIMEOUT_S
        logged_busy = False
        while not self._ctx.is_stopped() and not self._ctx.pause_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self._ctx.logger.behavior(
                    "[DANGER] input admission timeout — escape remains pending"
                )
                return False
            try:
                parameters = inspect.signature(begin).parameters.values()
                accepts_timeout = (
                    "timeout_s" in inspect.signature(begin).parameters
                    or any(
                        parameter.kind is inspect.Parameter.VAR_KEYWORD
                        for parameter in parameters
                    )
                )
            except (TypeError, ValueError):
                accepts_timeout = True
            admitted = (
                begin(timeout_s=remaining)
                if accepts_timeout
                else begin()
            )
            if admitted is not False:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self._ctx.logger.behavior(
                    "[DANGER] input admission timeout — escape remains pending"
                )
                return False
            if not logged_busy:
                self._ctx.logger.behavior(
                    "[DANGER] input busy after cancellation — waiting for escape admission"
                )
                logged_busy = True
            self._ctx.stop_event.wait(
                min(WORKER_POLL_INTERVAL_S, remaining)
            )
        return False

    def escape(self, *, seated: bool, reason: str) -> bool:
        """Run one complete escape or leave the fact active for a retry.

        Cancellation is deliberately issued *before* re-arming the input
        backend. ``begin_session`` acquires the same operation lock used by
        every keyboard/mouse operation; calling it first can fail while an
        attack/storage macro still owns that lock, which strands the danger
        transition before its teleport key is ever sent.
        """
        if self._ctx.is_stopped() or self._ctx.pause_event.is_set():
            return False
        with self._lock:
            if not self._ctx.begin_danger_transition(allow_sitting=seated):
                return False
            try:
                cancel = getattr(self._ctx, "cancel_gameplay_input", None)
                if callable(cancel):
                    cancel()
                if not self._begin_input_for_escape():
                    return False
                escaped = bool(
                    self._teleport.danger_teleport(
                        reason=reason,
                        prefer_safe_key=seated,
                    )
                )
                if not escaped:
                    self._ctx.logger.behavior(
                        f"[DANGER] {reason} teleport failed — retrying"
                    )
                    return False
                # Hunting danger is a normal teleport: the active hunt keeps
                # its area milestone, buffs, and timers (no new generation).
                # Only sit/stand recovery and kafra (storage) sessions break
                # the hunt loop. Sit danger remains inside the existing SP
                # session; its owner will call end_sit_ops after the
                # re-sit/recovery sequence.
                self._ctx.finish_danger_transition(seated=seated)
                self._ctx.logger.behavior(
                    f"[DANGER] {reason} escape complete"
                )
                return True
            except Exception as exc:
                self._ctx.logger.behavior(
                    f"[DANGER] {reason} transition error: {exc}"
                )
                return False
            finally:
                if self._ctx.danger_escape_active.is_set():
                    self._ctx.end_danger_transition()
