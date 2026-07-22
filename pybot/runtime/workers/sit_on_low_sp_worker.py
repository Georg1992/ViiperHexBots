"""Sit when SP is low; pause hunting (and timers) until SP recovers.

Before sitting: teleport until a discovery scan sees no living mobs, idle 1s,
then press sit. After SP recovers, stand, wait 500ms, then resume hunt/timers.

Each sit teleport clears tracking (same as hunt-mode teleport) so workers
resume against the new screen only.

SP comes from ``GameMemoryPoller`` → ``MemorySnapshot`` (memory addresses or
Basic Info vision — same ``sp`` / ``sp_max`` fields either way).
"""

from __future__ import annotations

from pybot.game_state import GameMemoryPoller
from pybot.config.clients import MemoryAddresses
from pybot.runtime.clear_area import HuntModeAreaReset, teleport_until_clear
from pybot.runtime.constants import (
    SIT_IDLE_BEFORE_SIT_S,
    SIT_LOW_SP_RATIO,
    SIT_RESUME_SP_RATIO,
    SIT_SP_POLL_INTERVAL_S,
    SIT_STAND_RESUME_DELAY_S,
)
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.workers.worker_contexts import SitOnLowSpWorkerContext


class SitOnLowSpWorker:
    """When SP drops below 5%, clear the area, sit until SP ≥ 98%, then stand."""

    def __init__(
        self,
        ctx: SitOnLowSpWorkerContext,
        input_backend: InputBackend,
        memory: MemoryAddresses,
        hunt_mode: HuntModeAreaReset,
        *,
        poller: GameMemoryPoller | None = None,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._memory = memory
        self._hunt_mode = hunt_mode
        self._poller = poller or GameMemoryPoller()
        self._last_fail_log = ""

    def run(self) -> None:
        ctx = self._ctx
        ctx.logger.behavior(
            f"[SIT] worker started key={ctx.config.sit_on_low_sp_button} "
            f"scanCode={ctx.config.sit_on_low_sp_scan_code} "
            f"low<{SIT_LOW_SP_RATIO:.0%} resume>={SIT_RESUME_SP_RATIO:.0%}"
        )
        while not ctx.is_stopped():
            try:
                if ctx.pause_event.is_set():
                    ctx.wait_while_stopped_or_paused(SIT_SP_POLL_INTERVAL_S)
                    continue
                ratio = self._sp_ratio()
                if ratio is None:
                    ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                    continue
                if ratio < SIT_LOW_SP_RATIO:
                    self._recover_sp(ratio)
                else:
                    ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
            except Exception:
                import traceback

                ctx.logger.behavior(f"[SIT] tick error:\n{traceback.format_exc()}")

    def _sp_ratio(self) -> float | None:
        ctx = self._ctx
        snap = self._poller.read(ctx.config.hwnd, self._memory)
        if not snap.ok or snap.sp is None or snap.sp_max is None or snap.sp_max <= 0:
            reason = snap.error or "sp_unavailable"
            if reason != self._last_fail_log:
                self._last_fail_log = reason
                ctx.logger.behavior(f"[SIT] SP read failed: {reason}")
            return None
        self._last_fail_log = ""
        return snap.sp / snap.sp_max

    def _recover_sp(self, low_ratio: float) -> None:
        ctx = self._ctx
        sit_scan = ctx.config.sit_on_low_sp_scan_code
        if not ctx.begin_sit_regen():
            return
        stood_up = False
        sat_down = False
        try:
            ctx.logger.behavior(
                f"[SIT] low SP ratio={low_ratio:.1%} — pausing hunt/timers, "
                "teleport until clear before sit"
            )
            if not teleport_until_clear(
                ctx, self._input, self._hunt_mode, log_tag="SIT"
            ):
                return

            ctx.logger.behavior(
                f"[SIT] area clear — idle {SIT_IDLE_BEFORE_SIT_S:.0f}s before sit"
            )
            if not ctx.wait_unless_stopped(SIT_IDLE_BEFORE_SIT_S):
                return

            self._input.teleport_key(sit_scan)
            sat_down = True

            while not ctx.is_stopped():
                ratio = self._sp_ratio()
                if ratio is not None and ratio >= SIT_RESUME_SP_RATIO:
                    ctx.logger.behavior(
                        f"[SIT] SP recovered ratio={ratio:.1%} — standing, "
                        f"wait {SIT_STAND_RESUME_DELAY_S * 1000:.0f}ms then resume"
                    )
                    self._input.teleport_key(sit_scan)
                    stood_up = True
                    ctx.wait_unless_stopped(SIT_STAND_RESUME_DELAY_S)
                    return
                ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)

            if sat_down and not stood_up:
                self._input.teleport_key(sit_scan)
                ctx.logger.behavior("[SIT] stopped while sitting — stood up")
        finally:
            ctx.end_sit_regen()
            ctx.discovery_wake.set()
