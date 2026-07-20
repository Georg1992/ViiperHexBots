"""Periodic skill timer key presses — one worker for all configured timers.

Timers press the key only (no mouse click), same as teleport_key.
"""

from __future__ import annotations

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
                    ctx.wait_while_stopped_or_paused(0.25)
                    continue

                now = monotonic_ms()
                next_wait_ms = 1000
                for timer in timers:
                    last = self._last_press_ms.get(timer.scan_code, 0)
                    elapsed = now - last
                    if elapsed >= timer.interval_ms:
                        self._input.teleport_key(timer.scan_code)
                        self._last_press_ms[timer.scan_code] = now
                        remaining = timer.interval_ms
                    else:
                        remaining = timer.interval_ms - elapsed
                    next_wait_ms = min(next_wait_ms, remaining)

                ctx.stop_event.wait(max(0.25, next_wait_ms / 1000.0))
            except Exception:
                import traceback

                ctx.logger.behavior(f"[TIMER] tick error:\n{traceback.format_exc()}")
