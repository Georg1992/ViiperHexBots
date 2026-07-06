"""Periodic skill timer key press — Reads the configured skill timer button and interval from the runtime
config and presses that key at the configured interval while the bot
is running in live mode.
"""

from __future__ import annotations

from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.workers.worker_contexts import SkillTimerWorkerContext


class SkillTimerWorker:
    """Presses the skill timer key at a fixed interval."""

    def __init__(
        self,
        ctx: SkillTimerWorkerContext,
        input_backend: InputBackend,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._last_press_ms = 0

    def run(self) -> None:
        ctx = self._ctx
        timer_scan_code = ctx.config.skill_timer_scan_code
        interval_ms = ctx.config.skill_timer_interval_ms

        if not timer_scan_code or interval_ms <= 0:
            return

        ctx.logger.behavior(
            f"[TIMER] skill timer started interval={interval_ms}ms "
            f"scanCode={timer_scan_code}"
        )

        while not ctx.is_stopped():
            if not ctx.should_run_workers():
                ctx.stop_event.wait(0.25)
                continue

            now = monotonic_ms()
            if now - self._last_press_ms >= interval_ms:
                self._input.skill_click(timer_scan_code)
                ctx.logger.behavior(
                    f"[TIMER] skill timer press scanCode={timer_scan_code}"
                )
                self._last_press_ms = now

            remaining = interval_ms - (now - self._last_press_ms)
            ctx.stop_event.wait(max(0.25, remaining / 1000.0))
