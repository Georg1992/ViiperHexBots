"""Periodic skill timer key presses — one worker for all configured timers.

Timers press the key only (no mouse click), same as teleport_key.
Paused while sitting/user-paused/teleporting. Teleport pause preserves
elapsed intervals; a new hunt generation (for example after sit recovery)
re-arms startup casts with ``SKILL_TIMER_STAGGER_MS`` staggering.
Storage sessions do not pause timers (combat only), so keys are not re-armed.
"""

from __future__ import annotations

import traceback

from pybot.runtime.constants import SKILL_TIMER_STAGGER_MS
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.input.input_backend import InputBackend, perform_if_allowed
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
        self._startup_generation: int | None = None
        self._startup_cycle_generation: int | None = None
        self._armed_generation: int | None = None
        self._startup_pressed: set[int] = set()

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
                generation = getattr(ctx, "hunt_generation", 0)
                if not self._wait_for_startup_buffs(generation):
                    if ctx.is_stopped():
                        break
                    continue
                if not ctx.should_run_timers():
                    if self._armed:
                        self._armed = False
                        ctx.logger.behavior("[TIMER] paused (sit/pause/danger/teleport)")
                    # Discovery suspension and a queued danger request do not
                    # stop ordinary workers, so the general pause wait would
                    # return immediately and spin. Use a bounded stop wait for
                    # those transient safety gates; use the resume gate for
                    # user pause/sit as before.
                    if ctx.should_run_workers():
                        ctx.stop_event.wait(0.25)
                    else:
                        ctx.wait_while_stopped_or_paused(0.25)
                    continue

                generation = int(getattr(ctx, "hunt_generation", 0))
                if self._startup_cycle_generation != generation:
                    # A sit/stand transition creates a new hunt generation.
                    # Re-arm immediately so every configured timer fires once
                    # for the new hunt instead of waiting for its old interval.
                    self._startup_pressed.clear()
                    self._startup_cycle_generation = generation
                    self._armed = False

                now = monotonic_ms()
                if not self._armed:
                    # Teleport suspension pauses the worker without creating a
                    # new hunt. Preserve elapsed timer time across that pause;
                    # only a genuinely new hunt generation re-arms timers from
                    # zero for its startup cast.
                    if self._armed_generation != generation:
                        self._arm_timers(timers)
                        self._armed_generation = generation
                    self._armed = True
                    ctx.logger.behavior("[TIMER] armed (hunt running)")
                    now = monotonic_ms()

                due = [
                    timer
                    for timer in timers
                    if now - self._last_press_ms.get(timer.scan_code, 0)
                    >= timer.interval_ms
                ]
                startup_pending = self._startup_generation != generation
                for timer in due:
                    if int(getattr(ctx, "hunt_generation", 0)) != generation:
                        break
                    if not ctx.should_run_timers():
                        break
                    if startup_pending and not self._startup_action_allowed():
                        break
                    if not self._wait_stagger_gap():
                        break
                    if int(getattr(ctx, "hunt_generation", 0)) != generation:
                        break
                    if not ctx.should_run_timers():
                        break
                    if startup_pending and not self._startup_action_allowed():
                        break
                    pressed = perform_if_allowed(
                        self._input,
                        ctx.should_run_timers,
                        lambda: self._input.teleport_key(timer.scan_code),
                        lifecycle=ctx,
                    )
                    if pressed is False:
                        continue
                    if int(getattr(ctx, "hunt_generation", 0)) != generation:
                        break
                    pressed_at = monotonic_ms()
                    self._last_press_ms[timer.scan_code] = pressed_at
                    self._last_any_press_ms = pressed_at
                    self._startup_pressed.add(timer.scan_code)

                # Release combat only after every normal timer has fired once
                # for this hunt generation. The generation changes when sit
                # ends, so the startup sequence repeats on the next hunt.
                generation = int(getattr(ctx, "hunt_generation", 0))
                startup_scans = {timer.scan_code for timer in timers}
                if (
                    self._startup_generation != generation
                    and startup_scans.issubset(self._startup_pressed)
                ):
                    mark_done = getattr(ctx, "mark_startup_timers_done", None)
                    if callable(mark_done):
                        completed = mark_done(expected_generation=generation)
                        if completed is False:
                            continue
                    self._startup_generation = generation

                now = monotonic_ms()
                next_wait_ms = 1000
                for timer in timers:
                    elapsed = now - self._last_press_ms.get(timer.scan_code, 0)
                    remaining = max(0, timer.interval_ms - elapsed)
                    next_wait_ms = min(next_wait_ms, remaining)

                ctx.stop_event.wait(max(0.05, next_wait_ms / 1000.0))
            except Exception:
                ctx.logger.behavior(f"[TIMER] tick error:\n{traceback.format_exc()}")

    def _startup_action_allowed(self) -> bool:
        checker = getattr(self._ctx, "should_run_startup_actions", None)
        if checker is None:
            return self._ctx.should_run_timers()
        return bool(checker())

    def _wait_for_startup_buffs(self, generation: int) -> bool:
        """Wait until character buffs finish before firing normal timers."""
        custom = getattr(self._ctx.config, "custom_behavior", None)
        buffs = getattr(custom, "buffs", ())
        if not buffs:
            mark_done = getattr(self._ctx, "mark_startup_buffs_done", None)
            if callable(mark_done):
                completed = mark_done(expected_generation=generation)
                if completed is False:
                    return False
            return True
        event = getattr(self._ctx, "startup_buffs_done", None)
        if event is None or event.is_set():
            return True
        while not self._ctx.is_stopped():
            if generation != getattr(self._ctx, "hunt_generation", generation):
                return False
            if not self._ctx.should_run_timers():
                # Teleport/danger suspension leaves ordinary workers runnable,
                # so the general pause wait would return immediately and spin.
                # User pause/sit still use the resume gate.
                should_run_workers = getattr(
                    self._ctx,
                    "should_run_workers",
                    None,
                )
                if callable(should_run_workers) and should_run_workers():
                    self._ctx.stop_event.wait(0.25)
                else:
                    self._ctx.wait_while_stopped_or_paused(0.25)
                continue
            if event.wait(0.05):
                return True
        return False

    def _arm_timers(self, timers) -> None:
        """Start timers for a new hunt so each configured key is due once."""
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
