"""Sit when SP is low; pause hunting (and timers) until SP recovers.

Before sitting: teleport until a discovery scan sees no living mobs, idle 1s,
then press sit. After SP recovers, stand and resume hunt.

Each sit teleport clears tracking (same as hunt-mode teleport) so workers
resume against the new screen only.

SP comes from ``GameMemoryPoller`` → ``MemorySnapshot`` (memory addresses or
Basic Info vision — same ``sp`` / ``sp_max`` fields either way).
"""

from __future__ import annotations

from typing import Protocol

from pybot.app.process_memory import GameMemoryPoller
from pybot.config.clients import MemoryAddresses
from pybot.runtime.constants import (
    SIT_IDLE_BEFORE_SIT_S,
    SIT_LOW_SP_RATIO,
    SIT_RESUME_SP_RATIO,
    SIT_SP_POLL_INTERVAL_S,
)
from pybot.runtime.detection.discovery_filter import filter_scan_candidates
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.workers.worker_contexts import SitOnLowSpWorkerContext


class _HuntModeAreaReset(Protocol):
    def on_area_reset(self) -> None: ...


class SitOnLowSpWorker:
    """When SP drops below 5%, clear the area, sit until SP ≥ 98%, then stand."""

    def __init__(
        self,
        ctx: SitOnLowSpWorkerContext,
        input_backend: InputBackend,
        memory: MemoryAddresses,
        hunt_mode: _HuntModeAreaReset,
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
        ctx.begin_sit_regen()
        stood_up = False
        sat_down = False
        try:
            ctx.logger.behavior(
                f"[SIT] low SP ratio={low_ratio:.1%} — pausing hunt/timers, "
                "teleport until clear before sit"
            )
            if not self._teleport_until_clear():
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
                        f"[SIT] SP recovered ratio={ratio:.1%} — standing, resume hunt"
                    )
                    self._input.teleport_key(sit_scan)
                    stood_up = True
                    return
                ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)

            if sat_down and not stood_up:
                self._input.teleport_key(sit_scan)
                ctx.logger.behavior("[SIT] stopped while sitting — stood up")
        finally:
            ctx.end_sit_regen()
            ctx.discovery_wake.set()

    def _teleport_until_clear(self) -> bool:
        """Teleport + settle until a discovery scan finds zero living mobs."""
        ctx = self._ctx
        tp_scan = ctx.config.teleport_scan_code
        if tp_scan <= 0:
            ctx.logger.behavior(
                "[SIT] no teleport key configured — cannot clear area before sit"
            )
            return False

        while not ctx.is_stopped():
            living = self._scan_living_count()
            if living is None:
                ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                continue
            if living == 0:
                ctx.logger.behavior("[SIT] discovery sees no mobs")
                self._reset_tracking_after_teleport("sit_clear")
                return True

            ctx.logger.behavior(
                f"[SIT] discovery living={living} — teleport before sit"
            )
            self._input.teleport_key(tp_scan)
            ctx.overlay.increment_teleports()
            delay_s = ctx.config.teleport_duration_ms / 1000.0
            if not ctx.wait_unless_stopped(delay_s):
                return False
            # New screen — drop pre-teleport tracks so hunt resumes clean.
            self._reset_tracking_after_teleport("sit_teleport")
        return False

    def _reset_tracking_after_teleport(self, reason: str) -> None:
        """Clear tracks/policy/overlay and hunt-mode area flags after a sit teleport.

        Tracking is paused while sitting, so the overlay would otherwise keep
        showing pre-teleport track markers until hunt workers resume.
        """
        ctx = self._ctx
        ctx.area_reset(reason)
        self._hunt_mode.on_area_reset()
        ctx.overlay.set_track_stats(track_count=0, alive_count=0)
        ctx.overlay.set_track_positions([])
        ctx.logger.behavior(f"[SIT] tracking reset reason={reason}")

    def _scan_living_count(self) -> int | None:
        """Run one discovery scan; return filtered living count or None on failure."""
        ctx = self._ctx
        if not ctx.capture.is_valid():
            return None
        roi = ctx.capture.get_hunt_roi()
        if roi is None:
            return None
        frame = ctx.capture.capture_roi(roi)
        if frame is None or frame.size == 0:
            return None
        scan = ctx.detector.discover_frame(frame, roi)
        if not scan.ok:
            return None
        filtered = filter_scan_candidates(
            scan.detections, roi, ctx.config.cell_size_px
        )
        return len(filtered)
