"""Sit when SP is low; pause hunting (and timers) until SP recovers.

Before sitting: teleport until discovery sees no mobs, then sit.
While sitting: if DangerDetector detects damage (HP drop), stand and
return ``"interrupted"`` so ``_recover_sp`` finds a new spot and sits again.
On SP recover: stand and resume.

Hunt never resumes until standing is confirmed. SP comes from shared
``PlayerVitals``. Danger detection is handled by ``DangerDetector``.
"""

from __future__ import annotations

import traceback

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import (
    SIT_LOW_SP_RATIO,
    SIT_POSE_SETTLE_S,
    SIT_RESUME_SP_RATIO,
    SIT_SP_POLL_INTERVAL_S,
    SIT_STAND_RESUME_DELAY_S,
)
from pybot.recognition.ui.character_pose import (
    measure_center_pose,
    check_is_sitting,
)
from pybot.runtime.danger_detector import DangerDetector
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.teleport import TeleportController
from pybot.runtime.workers.worker_contexts import SitOnLowSpWorkerContext

_SIT_VERIFY_RETRIES = 3


class SitOnLowSpWorker:
    """When SP drops below 5%, clear the area, sit until SP ≥ 98%, then stand."""

    def __init__(
        self,
        ctx: SitOnLowSpWorkerContext,
        input_backend: InputBackend,
        teleport: TeleportController,
        *,
        danger: DangerDetector,
        vitals: PlayerVitals,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._teleport = teleport
        self._vitals = vitals
        self._danger = danger
        self._last_fail_log = ""

    def _capture_frame(self):
        """Capture a hunt frame for pose verification."""
        roi = self._ctx.capture.get_hunt_roi()
        if roi is None:
            return None
        return self._ctx.capture.capture_roi(roi)

    def _pose_is_sitting(self) -> bool | None:
        """True if sitting, False if standing, None if pose cannot be read."""
        frame = self._capture_frame()
        if frame is None or frame.size == 0:
            return None
        return check_is_sitting(measure_center_pose(frame))

    def run(self) -> None:
        ctx = self._ctx
        ctx.logger.behavior(
            f"[SIT] worker started key={ctx.config.sit_on_low_sp_button} "
            f"scanCode={ctx.config.sit_on_low_sp_scan_code} "
            f"low<{SIT_LOW_SP_RATIO:.0%} resume>={SIT_RESUME_SP_RATIO:.0%}"
        )
        while not ctx.is_stopped():
            try:
                if not ctx.should_run_workers():
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
                ctx.logger.behavior(f"[SIT] tick error:\n{traceback.format_exc()}")

    def _sp_snapshot(self) -> tuple[int, float] | None:
        """Return ``(sp, ratio)`` or None when SP is unavailable."""
        ctx = self._ctx
        sp, sp_max = self._vitals.sp_pair()
        if sp is None or sp_max is None or sp_max <= 0:
            reason = "sp_unavailable"
            if reason != self._last_fail_log:
                self._last_fail_log = reason
                ctx.logger.behavior(f"[SIT] SP read failed: {reason}")
            return None
        self._last_fail_log = ""
        return sp, sp / sp_max

    def _sp_ratio(self) -> float | None:
        snap = self._sp_snapshot()
        return None if snap is None else snap[1]

    def _recover_sp(self, low_ratio: float) -> None:
        ctx = self._ctx
        sit_scan = ctx.config.sit_on_low_sp_scan_code
        if not ctx.begin_sit_ops():
            return
        try:
            ctx.logger.behavior(
                f"[SIT] low SP ratio={low_ratio:.1%} — pausing hunt/timers, "
                "teleport until clear before sit"
            )
            while not ctx.is_stopped():
                if ctx.pause_event.is_set():
                    if not ctx.wait_while_stopped_or_paused(SIT_SP_POLL_INTERVAL_S):
                        return
                    continue
                if not self._teleport.teleport_until_quiet(log_tag="SIT"):
                    return
                outcome = self._sit_session()
                if outcome == "recovered":
                    return
                if outcome is None:
                    return
                ctx.logger.behavior(
                    f"[SIT] {outcome} while regenerating — "
                    "finding another sit spot"
                )
        finally:
            # Never release the hunt gate while still sitting.
            self._stand_before_hunt_resume(sit_scan)
            ctx.end_sit_ops()
            ctx.discovery_wake.set()

    def _stand_before_hunt_resume(self, sit_scan: int) -> None:
        """Block until standing is confirmed (or stop), then hunt may resume."""
        ctx = self._ctx
        while not ctx.is_stopped():
            if self._ensure_standing(sit_scan, from_sit=True):
                return
            ctx.logger.behavior(
                "[SIT] standing not confirmed — retrying before hunt resume"
            )
            ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)

    def _sit_session(self) -> str | None:
        """Sit until SP recovers.

        Returns:
            ``"recovered"`` — stood after SP ≥ resume threshold.
            ``"interrupted"`` — damage while sitting; stood before return.
            ``None`` — stop/abort (could not sit, or stopped).
        """
        ctx = self._ctx
        sit_scan = ctx.config.sit_on_low_sp_scan_code

        # Reset the flag so we only detect damage WHILE sitting.
        self._danger.pop_damage_detected()

        if not self._ensure_sitting(sit_scan):
            ctx.logger.behavior(
                "[SIT] could not confirm sitting — aborting session"
            )
            return None
        ctx.logger.behavior("[SIT] sitting confirmed — waiting for regen")

        while not ctx.is_stopped():
            if ctx.pause_event.is_set():
                if not ctx.wait_while_stopped_or_paused(SIT_SP_POLL_INTERVAL_S):
                    return None
                continue

            if self._danger.pop_damage_detected():
                ctx.logger.behavior(
                    "[SIT] interrupted by damage — standing before new spot"
                )
                if not self._ensure_standing(sit_scan, from_sit=True):
                    ctx.logger.behavior(
                        "[SIT] could not confirm standing after damage — "
                        "holding sit gate"
                    )
                    ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                    continue
                return "interrupted"

            sp_state = self._sp_snapshot()
            if sp_state is not None:
                _sp, ratio = sp_state
                if ratio >= SIT_RESUME_SP_RATIO:
                    ctx.logger.behavior(
                        f"[SIT] SP recovered ratio={ratio:.1%} — standing"
                    )
                    if not self._ensure_standing(sit_scan, from_sit=True):
                        # Keep the sit gate held; retry stand without resuming hunt.
                        ctx.logger.behavior(
                            "[SIT] could not confirm standing — retrying"
                        )
                        ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                        continue
                    if not ctx.wait_unless_stopped(SIT_STAND_RESUME_DELAY_S):
                        return None
                    return "recovered"

            ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
        return None

    def _ensure_sitting(self, sit_scan: int) -> bool:
        """Reach sitting pose; press the toggle only when not already sitting."""
        return self._ensure_pose(sit_scan, want_sit=True)

    def _ensure_standing(self, sit_scan: int, *, from_sit: bool = False) -> bool:
        """Reach standing pose.

        When ``from_sit`` is True we always press the toggle once first. The
        falcon can make a sitting sprite look standing, which would otherwise
        skip the stand key and resume hunt while still seated.
        """
        if from_sit:
            self._input.key_tap(sit_scan)
            if not self._ctx.wait_unless_stopped(SIT_POSE_SETTLE_S):
                return False
            if self._pose_is_sitting() is False:
                return True
        return self._ensure_pose(sit_scan, want_sit=False)

    def _ensure_pose(self, sit_scan: int, *, want_sit: bool) -> bool:
        """Toggle-safe sit/stand: succeed only on a confirmed pose read.

        Unreadable/ambiguous pose is never treated as success. After each
        press we wait ``SIT_POSE_SETTLE_S`` before verifying.
        """
        label = "sit" if want_sit else "stand"
        for attempt in range(_SIT_VERIFY_RETRIES):
            if self._ctx.is_stopped():
                return False
            if self._ctx.pause_event.is_set():
                if not self._ctx.wait_while_stopped_or_paused(SIT_SP_POLL_INTERVAL_S):
                    return False
                continue
            sitting = self._pose_is_sitting()
            if sitting is not None and sitting == want_sit:
                return True
            self._input.key_tap(sit_scan)
            if not self._ctx.wait_unless_stopped(SIT_POSE_SETTLE_S):
                if self._ctx.is_stopped():
                    return False
                continue
            sitting = self._pose_is_sitting()
            if sitting is not None and sitting == want_sit:
                return True
            self._ctx.logger.behavior(
                f"[SIT] {label} verify failed "
                f"attempt {attempt + 1}/{_SIT_VERIFY_RETRIES} — retrying"
            )
        return False
