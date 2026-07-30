"""Sit when SP is low; pause hunting (and timers) until SP recovers.

Before sitting: teleport until a discovery scan sees no living mobs, idle 1s,
measure standing pose, sit and confirm the shorter sitting pose (retry on
failure), then regenerate. While regenerating: if something makes the character stand up (e.g.
DangerDetector teleport), the pose changes — detected and the sit session
returns ``"interrupted"`` so ``_recover_sp`` loops to find a new quiet spot.
On SP recover: stand and confirm standing pose before resume (retry on
failure).

Each sit teleport clears tracking (same as hunt-mode teleport) so workers
resume against the new screen only.

SP comes from shared ``PlayerVitals``. Danger detection (HP, critical, surrounded) is handled by
``DangerDetector`` — the sit worker only detects interruptions.
"""

from __future__ import annotations

import time

from pybot.game_state import PlayerVitals
from pybot.recognition.ui.character_pose import CharacterPose, measure_center_pose
from pybot.runtime.clear_area import HuntModeAreaReset, teleport_until_quiet
from pybot.runtime.constants import (
    SIT_POSE_CHECK_S,
    SIT_LOW_SP_RATIO,
    SIT_POSE_MAX_ATTEMPTS,
    SIT_POSE_MIN_HEIGHT_DROP,
    SIT_POSE_SETTLE_S,
    SIT_RESUME_SP_RATIO,
    SIT_SP_POLL_INTERVAL_S,
    SIT_STAND_RESUME_DELAY_S,
)
from pybot.runtime.danger_detector import DangerDetector
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.mob_behaviors import MobBehavior
from pybot.runtime.workers.worker_contexts import SitOnLowSpWorkerContext

_STAND_HEIGHT = 99  # standing pose body_height for interruption check


class SitOnLowSpWorker:
    """When SP drops below 5%, clear the area, sit until SP ≥ 98%, then stand."""

    def __init__(
        self,
        ctx: SitOnLowSpWorkerContext,
        input_backend: InputBackend,
        hunt_mode: HuntModeAreaReset,
        *,
        danger: DangerDetector | None = None,
        mob_behavior: MobBehavior | None = None,
        vitals: PlayerVitals | None = None,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._hunt_mode = hunt_mode
        self._mob_behavior = mob_behavior or MobBehavior()
        self._vitals = vitals or PlayerVitals()
        self._danger = danger or DangerDetector(
            ctx, input_backend, self._mob_behavior, vitals=self._vitals,
        )
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

    def _capture_client(self):
        frame = self._ctx.capture.capture_client()
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        return frame

    def _measure_pose(self) -> CharacterPose | None:
        frame = self._capture_client()
        if frame is None:
            return None
        return measure_center_pose(frame)

    def _recover_sp(self, low_ratio: float) -> None:
        ctx = self._ctx
        if not ctx.begin_sit_regen():
            return
        try:
            ctx.logger.behavior(
                f"[SIT] low SP ratio={low_ratio:.1%} — pausing hunt/timers, "
                "teleport until clear before sit"
            )
            while not ctx.is_stopped():
                # teleport_until_quiet scans the screen for a safe place
                # using the normal (creamy-first) teleport key.
                if not teleport_until_quiet(
                    ctx,
                    self._input,
                    self._hunt_mode,
                    log_tag="SIT",

                ):
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
            ctx.end_sit_regen()
            ctx.discovery_wake.set()

    def _sit_session(self) -> str:
        """Sit and wait for SP recovery.            Returns:
                ``"recovered"`` — stood after SP ≥ resume threshold.
                ``"interrupted"`` — no longer sitting (teleported, stood up, etc.).
        """
        ctx = self._ctx
        sit_scan = ctx.config.sit_on_low_sp_scan_code

        self._ensure_sitting(sit_scan)
        ctx.logger.behavior("[SIT] Pressed sit button, waiting for regen")

        sp_state = self._sp_snapshot()
        last_sp = sp_state[0] if sp_state is not None else None
        last_pose_check = 0.0

        while not ctx.is_stopped():
            sp_state = self._sp_snapshot()
            if sp_state is not None:
                sp, ratio = sp_state
                if ratio >= SIT_RESUME_SP_RATIO:
                    ctx.logger.behavior(
                        f"[SIT] SP recovered ratio={ratio:.1%} — standing"
                    )
                    self._ensure_standing(sit_scan)
                    ctx.wait_unless_stopped(SIT_STAND_RESUME_DELAY_S)
                    return "recovered"

                now = time.monotonic()
                if last_sp is None:
                    last_sp = sp
                elif sp > last_sp:
                    last_sp = sp

                # Periodic pose check: if standing when we should be sitting,
                # something interrupted us (teleport, manual, etc.) — retry.
                if now - last_pose_check >= SIT_POSE_CHECK_S:
                    last_pose_check = now
                    pose = self._measure_pose()
                    if pose is not None and pose.body_height >= _STAND_HEIGHT:
                        ctx.logger.behavior(
                            f"[SIT] interrupted while sitting sp={sp} — "
                            "standing pose detected"
                        )
                        return "interrupted"

            ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)

    def _ensure_sitting(
        self,
        sit_scan: int,
    ) -> None:
        """Press sit once."""
        self._input.teleport_key(sit_scan)
        self._ctx.wait_unless_stopped(SIT_POSE_SETTLE_S)

    def _ensure_standing(
        self,
        sit_scan: int,
    ) -> None:
        """Press stand once."""
        self._input.teleport_key(sit_scan)
        self._ctx.wait_unless_stopped(SIT_POSE_SETTLE_S)
