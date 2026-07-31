"""Periodic per-mob self-buff casts."""

from __future__ import annotations

import traceback

from pybot.runtime.constants import SKILL_TIMER_STAGGER_MS
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.workers.worker_contexts import SelfBuffWorkerContext


class SelfBuffWorker:
    """Cast configured buffs on the character once per configured delay."""

    def __init__(self, ctx: SelfBuffWorkerContext, input_backend: InputBackend) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._last_cast_ms: dict[int, int] = {}
        self._last_any_cast_ms = 0
        self._armed = False

    def run(self) -> None:
        ctx = self._ctx
        buffs = tuple(
            buff
            for buff in ctx.config.custom_behavior.buffs
            if buff.scan_code > 0 and buff.delay_ms > 0
        )
        if not buffs:
            return

        armed_at = monotonic_ms()
        for buff in buffs:
            self._last_cast_ms[id(buff)] = armed_at
            ctx.logger.behavior(
                f"[CUSTOM] buff started key={buff.button} "
                f"interval={buff.delay_ms}ms scanCode={buff.scan_code}"
            )

        while not ctx.is_stopped():
            try:
                if not ctx.should_run_combat():
                    self._armed = False
                    ctx.wait_while_combat_blocked(0.25)
                    continue

                now = monotonic_ms()
                if not self._armed:
                    # Resume each configured delay from the moment combat is
                    # available again; never fire immediately on gate release.
                    armed_at = monotonic_ms()
                    for buff in buffs:
                        self._last_cast_ms[id(buff)] = armed_at
                    self._armed = True
                    now = armed_at

                due = [
                    buff
                    for buff in buffs
                    if now - self._last_cast_ms[id(buff)] >= buff.delay_ms
                ]
                for buff in due:
                    if not ctx.should_run_combat():
                        break
                    if not self._wait_stagger_gap():
                        break
                    pos = ctx.character_screen_pos()
                    if pos is None or not ctx.should_run_combat():
                        break
                    cx, cy = int(pos[0]), int(pos[1])
                    if self._input.skill_click_at(buff.scan_code, cx, cy):
                        cast_at = monotonic_ms()
                        self._last_cast_ms[id(buff)] = cast_at
                        self._last_any_cast_ms = cast_at
                        ctx.logger.behavior(
                            f"[CUSTOM] buff cast key={buff.button} at=({cx},{cy})"
                        )
                    if ctx.is_stopped():
                        break

                if ctx.is_stopped():
                    break
                now = monotonic_ms()
                remaining = min(
                    max(0, item.delay_ms - (now - self._last_cast_ms[id(item)]))
                    for item in buffs
                )
                ctx.stop_event.wait(max(0.05, remaining / 1000.0))
            except Exception:
                ctx.logger.behavior(f"[CUSTOM] buff tick error:\n{traceback.format_exc()}")

    def _wait_stagger_gap(self) -> bool:
        ctx = self._ctx
        if self._last_any_cast_ms <= 0:
            return ctx.should_run_combat()
        gap = monotonic_ms() - self._last_any_cast_ms
        if gap >= SKILL_TIMER_STAGGER_MS:
            return True
        remaining_s = (SKILL_TIMER_STAGGER_MS - gap) / 1000.0
        if ctx.stop_event.wait(max(0.0, remaining_s)):
            return False
        return ctx.should_run_combat()
