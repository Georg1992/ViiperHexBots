"""Periodic per-mob self-buff casts on the character."""

from __future__ import annotations

import traceback

from pybot.runtime.constants import (
    LOG_REPEAT_INTERVAL_MS,
    STARTUP_BUFF_CURSOR_DELAY_S,
    STARTUP_BUFF_GAP_S,
)
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.input.input_backend import InputBackend, perform_if_allowed
from pybot.runtime.workers.worker_contexts import SelfBuffWorkerContext


class SelfBuffWorker:
    """Cast configured buffs on the character on each hunt cycle.

    Assigned character buffs cast first at each hunt start, in UI order,
    with a one-second gap between casts. Normal skill timers are released only
    after the full buff sequence completes. Each buff's periodic interval
    starts at its successful cast, and sitting starts a fresh hunt cycle.

    Buff casts share the :class:`~pybot.runtime.gate_controller.CharacterActionGate`
    with skill timers: a buff burst claims the shared keypress slot first so
    a buff and a timer due at the same instant never collide and the buff
    always fires before the timer.
    """

    def __init__(self, ctx: SelfBuffWorkerContext, input_backend: InputBackend) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._last_cast_ms: dict[int, int] = {}
        self._completed_generation: int | None = None
        self._last_error_log_ms: int | None = None

    def execute_buff(self, buff_key: int) -> bool:
        """Execute one periodic buff for the serialized scheduler."""
        ctx = self._ctx
        buffs = tuple(
            buff for buff in ctx.config.custom_behavior.buffs
            if buff.scan_code == buff_key and buff.button.strip() and buff.delay_ms > 0
        )
        if not buffs or ctx.is_stopped() or not ctx.should_run_character_actions():
            return False
        buff = buffs[0]
        if not self._wait_stagger_gap():
            return False
        if not self._cast_buff(buff):
            return False
        return True

    def process_pending(self) -> bool:
        """Advance startup buffs. Periodic casts are scheduler-owned."""
        ctx = self._ctx
        buffs = tuple(
            buff for buff in ctx.config.custom_behavior.buffs
            if buff.scan_code > 0 and buff.button.strip() and buff.delay_ms > 0
        )
        if not buffs or ctx.is_stopped():
            return False
        generation = self._current_generation()
        if self._completed_generation != generation:
            self._last_cast_ms.clear()
            if not self._run_startup_sequence(buffs, expected_generation=generation):
                return False
            if self._current_generation() != generation:
                return False
            mark = getattr(ctx, "mark_startup_buffs_done", None)
            if callable(mark) and mark(expected_generation=generation) is False:
                return False
            if not self._has_normal_timers():
                mark_timers = getattr(ctx, "mark_startup_timers_done", None)
                if callable(mark_timers):
                    mark_timers(expected_generation=generation)
            self._completed_generation = generation
            return True
        return False


    def last_success_ms(self, buff_key: int) -> int | None:
        """Return the last successful cast timestamp for scheduler seeding."""
        for buff in self._ctx.config.custom_behavior.buffs:
            if buff.scan_code == buff_key:
                return self._last_cast_ms.get(id(buff))
        return None

    def _current_generation(self) -> int:
        return int(getattr(self._ctx, "hunt_generation", 0))

    def _has_normal_timers(self) -> bool:
        return any(
            timer.scan_code and timer.button.strip() and timer.interval_ms > 0
            for timer in getattr(self._ctx.config, "skill_timers", ())
        )

    def _run_startup_sequence(
        self,
        buffs: tuple,
        *,
        expected_generation: int,
    ) -> bool:
        """Cast this generation's buffs in configured order.

        A generation change invalidates the sequence immediately. Completion
        is published by ``run`` only after this method returns successfully and
        the generation is still the same, preventing stale startup events from
        unlocking a later hunt.
        """
        for index, buff in enumerate(buffs):
            while not self._ctx.is_stopped():
                if self._critical_pending():
                    return False
                if self._current_generation() != expected_generation:
                    return False
                if not self._startup_action_allowed():
                    if self._critical_pending():
                        return False
                    self._ctx.wait_while_combat_blocked(0.25)
                    # In a recovered hunt's pre-clear window (danger escape /
                    # random fly-wing landing) combat is admitted, so
                    # wait_while_combat_blocked returns immediately, while
                    # startup actions stay blocked until the first discovery
                    # scan confirms the landing area. Re-check and pace the
                    # poll so the loop sleeps instead of spinning hot on the
                    # GIL and starving the very scan that would release it.
                    if not self._startup_action_allowed() and self._ctx.stop_event.wait(0.05):
                        return False
                    continue
                if not self._wait_stagger_gap(startup=True):
                    return False
                if self._cast_buff(buff, startup=True):
                    if self._current_generation() != expected_generation:
                        return False
                    break
                # Missing coordinates and transient input failures are
                # retryable and must not abandon the hunt-start sequence.
                if self._ctx.stop_event.wait(0.05):
                    return False
            else:
                return False

            if index + 1 < len(buffs) and not self._wait_startup_gap(
                expected_generation=expected_generation,
            ):
                return False
            if self._current_generation() != expected_generation:
                return False
        return self._current_generation() == expected_generation

    def _critical_pending(self) -> bool:
        """Return whether the pure danger observer requires preemption."""
        danger = getattr(self._ctx, "danger_detector", None)
        if danger is None:
            return False
        from pybot.runtime.danger_detector import DangerLevel
        return danger.danger_level() is DangerLevel.CRITICAL

    def _startup_action_allowed(self) -> bool:
        checker = getattr(self._ctx, "should_run_startup_actions", None)
        if checker is not None:
            return bool(checker())
        return self._character_action_allowed()

    def _character_action_allowed(self) -> bool:
        checker = getattr(self._ctx, "should_run_character_actions", None)
        if checker is not None:
            return bool(checker())
        return bool(self._ctx.should_run_combat())

    def _wait_startup_gap(self, *, expected_generation: int) -> bool:
        """Wait one full second while remaining in the active hunt."""
        deadline: int | None = None
        while not self._ctx.is_stopped():
            if self._critical_pending():
                return False
            if self._current_generation() != expected_generation:
                return False
            if not self._startup_action_allowed():
                # Ordinary combat blocking postpones the next buff. An urgent
                # critical request is different: abort this startup step so
                # GameplayLoop can consume the request immediately.
                if self._critical_pending():
                    return False
                deadline = None
                self._ctx.wait_while_combat_blocked(0.25)
                # Same re-check as _run_startup_sequence: combat may be
                # admitted before the area is confirmed clear; never spin.
                if not self._startup_action_allowed() and self._ctx.stop_event.wait(0.05):
                    return False
                continue
            if deadline is None:
                deadline = monotonic_ms() + int(STARTUP_BUFF_GAP_S * 1000)
            remaining_ms = deadline - monotonic_ms()
            if remaining_ms <= 0:
                return (
                    self._current_generation() == expected_generation
                    and self._startup_action_allowed()
                )
            if self._ctx.stop_event.wait(min(0.05, remaining_ms / 1000.0)):
                return False
        return False

    def _wait_stagger_gap(self, *, startup: bool = False) -> bool:
        """Wait for the shared character-action slot before a buff cast.

        Startup buffs and periodic buffs use the same gate as skill timers. The
        only difference is their lifecycle admission predicate: startup buffs
        may run before combat is released, while periodic buffs require normal
        combat safety.
        """
        gate = self._ctx.character_action_gate
        allowed = self._startup_action_allowed if startup else self._character_action_allowed
        while not self._ctx.is_stopped():
            if self._critical_pending():
                return False
            if not allowed():
                if self._critical_pending():
                    return False
                self._ctx.wait_while_combat_blocked(0.25)
                # Same re-check as the startup sequence: in the pre-clear
                # window combat is admitted while the action stays blocked.
                if not allowed() and self._ctx.stop_event.wait(0.05):
                    return False
                continue
            now = monotonic_ms()
            if gate.try_claim(is_buff=True, now_ms=now):
                return True
            remaining_ms = gate.stagger_remaining_ms(now)
            if remaining_ms <= 0:
                # A timer owns the stagger window; poll until it reopens.
                if self._ctx.stop_event.wait(0.05):
                    return False
                continue
            if self._ctx.stop_event.wait(remaining_ms / 1000.0):
                return False
        return False

    def _cast_buff(self, buff, *, startup: bool = False) -> bool:
        ctx = self._ctx
        try:
            allowed = self._startup_action_allowed() if startup else self._character_action_allowed()
            if not allowed:
                return False
            pos = ctx.character_screen_pos()
            if pos is None:
                return False
            cx, cy = int(pos[0]), int(pos[1])
            if startup:
                allowed = self._startup_action_allowed()
            else:
                allowed = self._character_action_allowed()
            if not allowed:
                return False
            def cast_action() -> bool:
                if startup:
                    try:
                        return bool(self._input.skill_click_at(
                            buff.scan_code,
                            cx,
                            cy,
                            move_delay_s=STARTUP_BUFF_CURSOR_DELAY_S,
                        ))
                    except TypeError:
                        # Keep lightweight test/custom backends compatible with
                        # the older three-argument protocol.
                        return bool(self._input.skill_click_at(buff.scan_code, cx, cy))
                return bool(self._input.skill_click_at(buff.scan_code, cx, cy))

            cast = perform_if_allowed(
                self._input,
                self._startup_action_allowed if startup else self._character_action_allowed,
                cast_action,
                lifecycle=ctx,
            )
            if not cast:
                return False
            cast_at = monotonic_ms()
            self._last_cast_ms[id(buff)] = cast_at
            # Record on the shared gate so the next buff cast and any timer
            # press both wait out the stagger window from this keypress.
            ctx.character_action_gate.note_action(cast_at)
            ctx.logger.behavior(
                f"[CUSTOM] buff cast key={buff.button} at=({cx},{cy})"
            )
            return True
        except Exception:
            self._log_error(
                f"[CUSTOM] buff cast failed key={buff.button}:\n"
                f"{traceback.format_exc()}"
            )
            return False

    def _log_error(self, message: str) -> None:
        """Throttle repeated custom-worker tracebacks under persistent failure."""
        now = monotonic_ms()
        if (
            self._last_error_log_ms is None
            or now - self._last_error_log_ms >= LOG_REPEAT_INTERVAL_MS
        ):
            self._last_error_log_ms = now
            self._ctx.logger.behavior(message)
