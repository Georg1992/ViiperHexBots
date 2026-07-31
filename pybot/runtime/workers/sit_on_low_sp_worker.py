"""Sit when SP is low; pause hunting (and timers) until SP recovers.

Before sitting: teleport until discovery sees no mobs, then sit.
While sitting: if DangerDetector detects damage (HP drop), stand and
return ``"interrupted"`` so ``_recover_sp`` finds a new spot and sits again.
On SP recover: stand and resume.

SP comes from shared ``PlayerVitals``. Danger detection is handled by
``DangerDetector`` — the sit worker only checks a flag.
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
    check_is_standing,
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

    def _verify_pose(self, expected_sit: bool) -> bool:
        """Capture frame, measure pose, return True if matching expected."""
        frame = self._capture_frame()
        if frame is None or frame.size == 0:
            return False
        pose = measure_center_pose(frame)
        if pose is None:
            return False
        if expected_sit:
            return bool(check_is_sitting(pose))
        return bool(check_is_standing(pose))

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
            ctx.end_sit_ops()
            ctx.discovery_wake.set()

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
                self._ensure_standing(sit_scan)
                return "interrupted"

            sp_state = self._sp_snapshot()
            if sp_state is not None:
                _sp, ratio = sp_state
                if ratio >= SIT_RESUME_SP_RATIO:
                    ctx.logger.behavior(
                        f"[SIT] SP recovered ratio={ratio:.1%} — standing"
                    )
                    if not self._ensure_standing(sit_scan):
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

    def _ensure_standing(self, sit_scan: int) -> bool:
        """Reach standing pose; press the toggle only when not already standing."""
        return self._ensure_pose(sit_scan, want_sit=False)

    def _ensure_pose(self, sit_scan: int, *, want_sit: bool) -> bool:
        """Toggle-safe sit/stand: verify first, press only on mismatch."""
        label = "sit" if want_sit else "stand"
        for attempt in range(_SIT_VERIFY_RETRIES):
            if self._ctx.is_stopped():
                return False
            if self._ctx.pause_event.is_set():
                if not self._ctx.wait_while_stopped_or_paused(SIT_SP_POLL_INTERVAL_S):
                    return False
                continue
            if self._verify_pose(expected_sit=want_sit):
                return True
            self._input.key_tap(sit_scan)
            if not self._ctx.wait_unless_stopped(SIT_POSE_SETTLE_S):
                if self._ctx.is_stopped():
                    return False
                continue
            if self._verify_pose(expected_sit=want_sit):
                return True
            self._ctx.logger.behavior(
                f"[SIT] {label} verify failed "
                f"attempt {attempt + 1}/{_SIT_VERIFY_RETRIES} — retrying"
            )
        return False
