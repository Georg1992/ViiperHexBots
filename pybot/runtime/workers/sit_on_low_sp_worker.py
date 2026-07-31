"""Sit when SP is low; pause hunting until SP recovers.

Contract
--------
* Hunt stays paused (``begin_sit_ops``) until SP ≥ resume threshold **or** stop.
  Aborting a sit attempt / failed area-clear must **retry**, never resume hunt.
* ``_seated`` tracks a confirmed sit. ``stand()`` presses once to leave sit;
  a second press only if pose is confirmed still sitting (toggle-safe).
* ``end_sit_ops`` runs only after the recover loop exits (SP recovered or stop).
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

_POSE_READS = 6


class SitOnLowSpWorker:
    """When SP < 5%: clear area, sit until SP ≥ 98%, stand, resume hunt."""

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
        self._seated = False

    def _pose_is_sitting(self) -> bool | None:
        """True=sitting, False=standing, None=unreadable/ambiguous."""
        roi = self._ctx.capture.get_hunt_roi()
        if roi is None:
            return None
        frame = self._ctx.capture.capture_roi(roi)
        if frame is None or frame.size == 0:
            return None
        return check_is_sitting(measure_center_pose(frame))

    def _tap(self, sit_scan: int, *, why: str) -> bool:
        self._ctx.logger.behavior(f"[SIT] tap sit-key reason={why}")
        self._input.key_tap(sit_scan)
        return self._ctx.wait_unless_stopped(SIT_POSE_SETTLE_S)

    def _ok_to_act(self) -> bool:
        ctx = self._ctx
        if ctx.is_stopped():
            return False
        while ctx.pause_event.is_set():
            if ctx.is_stopped():
                return False
            ctx.wait_while_user_paused(SIT_SP_POLL_INTERVAL_S)
        return not ctx.is_stopped()

    def sit(self, sit_scan: int) -> bool:
        """Confirm sitting. Press only when pose shows standing."""
        for _ in range(_POSE_READS):
            if not self._ok_to_act():
                return False
            sitting = self._pose_is_sitting()
            if sitting is True:
                self._seated = True
                self._ctx.logger.behavior("[SIT] seated confirmed")
                return True
            if sitting is False:
                if not self._tap(sit_scan, why="enter_sit"):
                    return False
                continue
            self._ctx.stop_event.wait(SIT_POSE_SETTLE_S)
        return False

    def stand(self, sit_scan: int) -> bool:
        """Leave sit: one press, re-press only if still confirmed sitting."""
        if not self._seated:
            return True
        if not self._ok_to_act():
            return False
        if not self._tap(sit_scan, why="leave_sit"):
            return False

        extra_presses = 0
        for _ in range(_POSE_READS):
            if not self._ok_to_act():
                return False
            sitting = self._pose_is_sitting()
            if sitting is False:
                self._seated = False
                self._ctx.logger.behavior("[SIT] standing confirmed")
                return True
            if sitting is True and extra_presses < 1:
                if not self._tap(sit_scan, why="still_sitting"):
                    return False
                extra_presses += 1
                continue
            self._ctx.stop_event.wait(SIT_POSE_SETTLE_S)
        self._ctx.logger.behavior("[SIT] standing NOT confirmed — keeping sit gate")
        return False

    def _ensure_standing_before_hunt(self, sit_scan: int) -> None:
        """Block until standing is confirmed (or stop)."""
        ctx = self._ctx
        while self._seated and not ctx.is_stopped():
            if self.stand(sit_scan):
                return
            ctx.logger.behavior("[SIT] retry stand before hunt resume")
            ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)

    def _sp_ratio(self) -> float | None:
        sp, sp_max = self._vitals.sp_pair()
        if sp is None or sp_max is None or sp_max <= 0:
            if self._last_fail_log != "sp_unavailable":
                self._last_fail_log = "sp_unavailable"
                self._ctx.logger.behavior("[SIT] SP read failed: sp_unavailable")
            return None
        self._last_fail_log = ""
        return sp / sp_max

    def _sp_recovered(self) -> bool:
        ratio = self._sp_ratio()
        return ratio is not None and ratio >= SIT_RESUME_SP_RATIO

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

    def _recover_sp(self, low_ratio: float) -> None:
        """Hold hunt until SP recovers. Failed sit/clear attempts retry in-place."""
        ctx = self._ctx
        sit_scan = ctx.config.sit_on_low_sp_scan_code
        self._seated = False
        if not ctx.begin_sit_ops():
            return
        try:
            ctx.logger.behavior(
                f"[SIT] low SP ratio={low_ratio:.1%} — hunt paused until "
                f"SP>={SIT_RESUME_SP_RATIO:.0%}"
            )
            while not ctx.is_stopped():
                if not self._ok_to_act():
                    break

                if not self._teleport.teleport_until_quiet(log_tag="SIT"):
                    ctx.logger.behavior(
                        "[SIT] area clear failed — retry (hunt stays paused)"
                    )
                    ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                    continue

                outcome = self._sit_until_done(sit_scan)
                if outcome == "recovered" and self._sp_recovered():
                    ratio = self._sp_ratio()
                    ctx.logger.behavior(
                        f"[SIT] SP recovered ratio={ratio:.1%} — resuming hunt"
                    )
                    break

                if outcome == "recovered" and not self._sp_recovered():
                    ctx.logger.behavior(
                        "[SIT] stood but SP still low — sit again "
                        "(hunt stays paused)"
                    )
                    continue

                if outcome == "interrupted":
                    ctx.logger.behavior(
                        "[SIT] interrupted — new sit spot (hunt stays paused)"
                    )
                    continue

                ctx.logger.behavior(
                    "[SIT] session incomplete — retry (hunt stays paused)"
                )
                ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
        finally:
            self._ensure_standing_before_hunt(sit_scan)
            # Loop only exits on SP recovered or stop — then release hunt.
            ctx.end_sit_ops()
            ctx.discovery_wake.set()

    def _sit_until_done(self, sit_scan: int) -> str | None:
        """Sit → wait for SP or damage → stand.

        Returns ``"recovered"`` only after standing with SP still ≥ resume.
        """
        ctx = self._ctx
        self._danger.pop_damage_detected()

        if not self.sit(sit_scan):
            ctx.logger.behavior("[SIT] could not confirm sitting — will retry")
            return None
        ctx.logger.behavior("[SIT] waiting for regen")

        while not ctx.is_stopped():
            if not self._ok_to_act():
                return None

            if self._danger.pop_damage_detected():
                ctx.logger.behavior("[SIT] damage — standing before new spot")
                if not self.stand(sit_scan):
                    ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                    continue
                return "interrupted"

            ratio = self._sp_ratio()
            if ratio is not None and ratio >= SIT_RESUME_SP_RATIO:
                ctx.logger.behavior(
                    f"[SIT] SP threshold met ratio={ratio:.1%} — standing"
                )
                if not self.stand(sit_scan):
                    ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                    continue
                if not ctx.wait_unless_stopped(SIT_STAND_RESUME_DELAY_S):
                    return None
                # Require SP still recovered after standing.
                if not self._sp_recovered():
                    ctx.logger.behavior(
                        "[SIT] SP dropped below resume after stand — not done"
                    )
                    return None
                return "recovered"

            ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
        return None
