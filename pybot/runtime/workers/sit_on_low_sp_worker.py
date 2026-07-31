"""Sit when SP is low; pause hunting until SP recovers.

State machine (one flag, two actions)::

    _seated=False
        │ sit()          → confirm sitting → _seated=True
        ▼
    _seated=True         (regen / wait)
        │ stand()        → press toggle → confirm standing → _seated=False
        ▼
    _seated=False        → end_sit_ops → hunt resumes

Rules:
* ``sit()`` may skip the key if pose already shows sitting.
* ``stand()`` runs only while ``_seated``; it always presses once first
  (falcon can fake a standing read), then verifies.
* ``finally`` calls ``stand()`` — a no-op when already stood, so there is
  never a second toggle that re-sits before hunt.
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

_POSE_RETRIES = 3


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

    # ── Pose I/O ─────────────────────────────────────────────────

    def _pose_is_sitting(self) -> bool | None:
        """True=sitting, False=standing, None=unreadable."""
        roi = self._ctx.capture.get_hunt_roi()
        if roi is None:
            return None
        frame = self._ctx.capture.capture_roi(roi)
        if frame is None or frame.size == 0:
            return None
        return check_is_sitting(measure_center_pose(frame))

    def _tap_sit_key(self, sit_scan: int) -> bool:
        """Press sit/stand toggle and wait for the pose animation."""
        self._input.key_tap(sit_scan)
        return self._ctx.wait_unless_stopped(SIT_POSE_SETTLE_S)

    def _ok_to_act(self) -> bool:
        """False if stopped. Waits out user pause while sitting."""
        ctx = self._ctx
        if ctx.is_stopped():
            return False
        while ctx.pause_event.is_set():
            if ctx.is_stopped():
                return False
            ctx.wait_while_user_paused(SIT_SP_POLL_INTERVAL_S)
        return not ctx.is_stopped()

    # ── Sit / stand (the only toggle paths) ──────────────────────

    def sit(self, sit_scan: int) -> bool:
        """Become sitting. Sets ``_seated`` only after a confirmed sit read."""
        for attempt in range(_POSE_RETRIES):
            if not self._ok_to_act():
                return False
            if self._pose_is_sitting() is True:
                self._seated = True
                return True
            if not self._tap_sit_key(sit_scan):
                return False
            if self._pose_is_sitting() is True:
                self._seated = True
                return True
            self._ctx.logger.behavior(
                f"[SIT] sit verify failed attempt {attempt + 1}/{_POSE_RETRIES}"
            )
        return False

    def stand(self, sit_scan: int) -> bool:
        """Become standing if seated. No-op when ``_seated`` is already False.

        Always presses before the first verify — we know we sat, so a falcon
        false-standing read must not skip the key.
        """
        if not self._seated:
            return True
        for attempt in range(_POSE_RETRIES):
            if not self._ok_to_act():
                return False
            if not self._tap_sit_key(sit_scan):
                return False
            if self._pose_is_sitting() is False:
                self._seated = False
                return True
            self._ctx.logger.behavior(
                f"[SIT] stand verify failed attempt {attempt + 1}/{_POSE_RETRIES}"
            )
        return False

    # ── SP helpers ───────────────────────────────────────────────

    def _sp_ratio(self) -> float | None:
        sp, sp_max = self._vitals.sp_pair()
        if sp is None or sp_max is None or sp_max <= 0:
            if self._last_fail_log != "sp_unavailable":
                self._last_fail_log = "sp_unavailable"
                self._ctx.logger.behavior("[SIT] SP read failed: sp_unavailable")
            return None
        self._last_fail_log = ""
        return sp / sp_max

    # ── Worker loop ──────────────────────────────────────────────

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
        ctx = self._ctx
        sit_scan = ctx.config.sit_on_low_sp_scan_code
        self._seated = False
        if not ctx.begin_sit_ops():
            return
        try:
            ctx.logger.behavior(
                f"[SIT] low SP ratio={low_ratio:.1%} — pausing hunt/timers, "
                "teleport until clear before sit"
            )
            while not ctx.is_stopped():
                if not self._ok_to_act():
                    return
                if not self._teleport.teleport_until_quiet(log_tag="SIT"):
                    return
                outcome = self._sit_until_done(sit_scan)
                if outcome == "recovered":
                    return
                if outcome is None:
                    return
                ctx.logger.behavior(
                    f"[SIT] {outcome} — finding another sit spot"
                )
        finally:
            self.stand(sit_scan)  # no-op if already stood
            ctx.end_sit_ops()
            ctx.discovery_wake.set()

    def _sit_until_done(self, sit_scan: int) -> str | None:
        """Sit, regen, stand.

        Returns ``"recovered"``, ``"interrupted"`` (stood after damage),
        or ``None`` (abort / stop).
        """
        ctx = self._ctx
        self._danger.pop_damage_detected()

        if not self.sit(sit_scan):
            ctx.logger.behavior("[SIT] could not confirm sitting — aborting")
            return None
        ctx.logger.behavior("[SIT] sitting confirmed — waiting for regen")

        while not ctx.is_stopped():
            if not self._ok_to_act():
                return None

            if self._danger.pop_damage_detected():
                ctx.logger.behavior("[SIT] damage — standing before new spot")
                if not self.stand(sit_scan):
                    ctx.logger.behavior("[SIT] stand failed after damage — retry")
                    ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                    continue
                return "interrupted"

            ratio = self._sp_ratio()
            if ratio is not None and ratio >= SIT_RESUME_SP_RATIO:
                ctx.logger.behavior(
                    f"[SIT] SP recovered ratio={ratio:.1%} — standing"
                )
                if not self.stand(sit_scan):
                    ctx.logger.behavior("[SIT] stand failed — retry")
                    ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                    continue
                if not ctx.wait_unless_stopped(SIT_STAND_RESUME_DELAY_S):
                    return None
                return "recovered"

            ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
        return None
