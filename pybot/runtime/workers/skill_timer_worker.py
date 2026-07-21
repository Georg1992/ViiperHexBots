"""Periodic skill timer key presses — one worker for all configured timers.

Timers press the key only (no mouse click), same as teleport_key.
Paused while sitting/user-paused; on hunt resume they re-arm and fire again
with ``SKILL_TIMER_STAGGER_MS`` between presses when several are due.
"""

from __future__ import annotations

from pybot.runtime.constants import SKILL_TIMER_STAGGER_MS
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.workers.worker_contexts import SkillTimerWorkerContext


class SkillTimerWorker:
    """Presses each skill-timer key at its own interval (key only, no click)."""

    def __init__(
        self,
        ctx: SkillTimerWorkerContext,
        input_backend: InputBackend,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._last_press_ms: dict[int, int] = {}
        self._last_any_press_ms = 0
        self._armed = False

    def run(self) -> None:
        ctx = self._ctx
        timers = [
            t
            for t in ctx.config.skill_timers
            if t.scan_code and t.interval_ms > 0
        ]
        if not timers:
            return

        for timer in timers:
            ctx.logger.behavior(
                f"[TIMER] started key={timer.button} interval={timer.interval_ms}ms "
                f"scanCode={timer.scan_code}"
            )
            self._last_press_ms[timer.scan_code] = 0

        while not ctx.is_stopped():
            try:
                if not ctx.should_run_workers():
                    if self._armed:
                        self._armed = False
                        ctx.logger.behavior("[TIMER] paused (sit/pause)")
                    ctx.wait_while_stopped_or_paused(0.25)
                    continue

                now = monotonic_ms()
                if not self._armed:
                    self._arm_timers(timers)
                    self._armed = True
                    ctx.logger.behavior("[TIMER] armed (hunt running)")
                    now = monotonic_ms()

                due = [
                    timer
                    for timer in timers
                    if now - self._last_press_ms.get(timer.scan_code, 0)
                    >= timer.interval_ms
                ]
                for timer in due:
                    if not ctx.should_run_workers():
                        break
                    if not self._wait_stagger_gap():
                        break
                    if not ctx.should_run_workers():
                        break
                    self._input.teleport_key(timer.scan_code)
                    pressed_at = monotonic_ms()
                    self._last_press_ms[timer.scan_code] = pressed_at
                    self._last_any_press_ms = pressed_at

                now = monotonic_ms()
                next_wait_ms = 1000
                for timer in timers:
                    elapsed = now - self._last_press_ms.get(timer.scan_code, 0)
                    remaining = max(0, timer.interval_ms - elapsed)
                    next_wait_ms = min(next_wait_ms, remaining)

                ctx.stop_event.wait(max(0.05, next_wait_ms / 1000.0))
            except Exception:
                import traceback

                ctx.logger.behavior(f"[TIMER] tick error:\n{traceback.format_exc()}")

    def _arm_timers(self, timers) -> None:
        """Start or restart all timers so they are due immediately (staggered)."""
        for timer in timers:
            self._last_press_ms[timer.scan_code] = 0
        self._last_any_press_ms = 0

    def _wait_stagger_gap(self) -> bool:
        """Ensure ``SKILL_TIMER_STAGGER_MS`` since the last timer press.

        Returns False if hunt stopped/paused/sitting before the gap elapsed.
        """
        ctx = self._ctx
        if self._last_any_press_ms <= 0:
            return ctx.should_run_workers()
        now = monotonic_ms()
        gap = now - self._last_any_press_ms
        if gap >= SKILL_TIMER_STAGGER_MS:
            return True
        return ctx.wait_while_stopped_or_paused(
            (SKILL_TIMER_STAGGER_MS - gap) / 1000.0,
        )
