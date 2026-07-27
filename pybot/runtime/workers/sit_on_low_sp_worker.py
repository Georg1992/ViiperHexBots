"""Sit when SP is low; pause hunting (and timers) until SP recovers.

Before sitting: teleport until a discovery scan sees no living mobs, idle 1s,
measure standing pose, sit and confirm the shorter sitting pose (retry on
failure), then regenerate. While regenerating: interrupt on danger — HP
dropping (vision; already standing from the hit), or SP stall/drop with
nearby foreign sprites (stand key with pose confirm if still seated) — then
teleport to a quiet area and sit again. On SP recover: stand and confirm
standing pose before resume (retry on failure).

Each sit teleport clears tracking (same as hunt-mode teleport) so workers
resume against the new screen only.

SP comes from shared ``PlayerVitals`` (UI publishes memory or Basic Info OCR).
HP for danger is always vision (status panel).
"""

from __future__ import annotations

import time

from pybot.game_state import PlayerVitals
from pybot.recognition.danger import DangerReport, assess_danger
from pybot.recognition.ui.character_pose import CharacterPose, measure_center_pose
from pybot.recognition.ui.status_panel import read_status_panel
from pybot.runtime.clear_area import HuntModeAreaReset, teleport_until_quiet
from pybot.runtime.constants import (
    SIT_HP_POLL_S,
    SIT_LOW_SP_RATIO,
    SIT_POSE_MAX_ATTEMPTS,
    SIT_POSE_MIN_HEIGHT_DROP,
    SIT_POSE_SETTLE_S,
    SIT_RESUME_SP_RATIO,
    SIT_SP_POLL_INTERVAL_S,
    SIT_SP_STALL_S,
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
        hunt_mode: HuntModeAreaReset,
        *,
        vitals: PlayerVitals | None = None,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._hunt_mode = hunt_mode
        self._vitals = vitals or PlayerVitals()
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

    def _read_hp(self, frame) -> int | None:
        """OCR current HP from Basic Info (vision-only)."""
        values = read_status_panel(frame)
        return None if values is None else values.hp

    def _assess_danger(
        self,
        frame,
        *,
        hp: int | None = None,
        previous_hp: int | None = None,
    ) -> DangerReport:
        return assess_danger(
            frame,
            cell_size_px=int(self._ctx.config.cell_size_px),
            hp=hp,
            previous_hp=previous_hp,
        )

    def _recover_sp(self, low_ratio: float) -> None:
        ctx = self._ctx
        if not ctx.begin_sit_regen():
            return
        try:
            ctx.logger.behavior(
                f"[SIT] low SP ratio={low_ratio:.1%} — pausing hunt/timers, "
                "teleport until clear before sit"
            )
            leave_screen = False
            while not ctx.is_stopped():
                # After sit danger (HP drop from unseen mobs), always TP once
                # before the discovery clear loop — otherwise 0 detections skip TP.
                if not teleport_until_quiet(
                    ctx,
                    self._input,
                    self._hunt_mode,
                    log_tag="SIT",
                    force_first=leave_screen,
                ):
                    return
                outcome = self._sit_session()
                if outcome == "recovered":
                    return
                if outcome == "stopped":
                    return
                leave_screen = True
                ctx.logger.behavior(
                    f"[SIT] {outcome} while regenerating — "
                    "force teleport, then find another sit spot"
                )
        finally:
            ctx.end_sit_regen()
            ctx.discovery_wake.set()

    def _sit_session(self) -> str:
        """Sit and wait for SP recovery.

        Returns:
            ``"recovered"`` — stood after SP ≥ resume threshold.
            ``"danger"`` — HP drop or SP stall with nearby objects.
            ``"pose_failed"`` — could not confirm sit/stand pose after retries.
            ``"stopped"`` — stop/pause ended the session (stood if needed).
        """
        ctx = self._ctx
        sit_scan = ctx.config.sit_on_low_sp_scan_code

        stand_pose = self._measure_stand_pose()
        if stand_pose is None:
            ctx.logger.behavior("[SIT] cannot measure stand pose — abort sit")
            return "pose_failed"

        sit_pose = self._ensure_sitting(sit_scan, stand_pose)
        if sit_pose is None:
            ctx.logger.behavior("[SIT] failed to confirm sit pose — abort sit")
            return "pose_failed"

        ctx.logger.behavior(
            f"[SIT] pose confirmed stand_h={stand_pose.body_height} "
            f"sit_h={sit_pose.body_height}"
        )

        sp_state = self._sp_snapshot()
        last_sp = sp_state[0] if sp_state is not None else None
        last_progress = time.monotonic()
        last_hp: int | None = None
        last_hp_poll = 0.0

        while not ctx.is_stopped():
            sp_state = self._sp_snapshot()
            if sp_state is not None:
                sp, ratio = sp_state
                if ratio >= SIT_RESUME_SP_RATIO:
                    ctx.logger.behavior(
                        f"[SIT] SP recovered ratio={ratio:.1%} — standing"
                    )
                    if not self._ensure_standing(sit_scan, sit_pose, stand_pose):
                        ctx.logger.behavior(
                            "[SIT] failed to confirm stand pose after recover, but SP is full so continuing"
                        )
                    else:
                        ctx.wait_unless_stopped(SIT_STAND_RESUME_DELAY_S)
                    return "recovered"

                now = time.monotonic()
                sp_stalled = False
                if last_sp is None:
                    last_sp = sp
                    last_progress = now
                elif sp > last_sp:
                    last_sp = sp
                    last_progress = now
                elif sp < last_sp:
                    if now - last_progress >= SIT_SP_STALL_S:
                        sp_stalled = True
                elif now - last_progress >= SIT_SP_STALL_S:
                    sp_stalled = True

                # HP OCR on its own cadence (always vision).
                if now - last_hp_poll >= SIT_HP_POLL_S:
                    frame = self._capture_client()
                    last_hp_poll = now
                    if frame is not None:
                        hp = self._read_hp(frame)
                        danger = self._assess_danger(
                            frame, hp=hp, previous_hp=last_hp
                        )
                        if danger.hp_dropped:
                            ctx.logger.behavior(
                                f"[SIT] danger while sitting sp={sp} "
                                f"reasons={','.join(danger.reasons)} "
                                "(hp dropped, ensuring standing)"
                            )
                            self._ensure_standing(
                                sit_scan, sit_pose, stand_pose
                            )
                            return "danger"
                        if (
                            sp_stalled
                            and danger.has_near_objects
                        ):
                            ctx.logger.behavior(
                                f"[SIT] danger while sitting sp={sp} "
                                f"reasons={','.join(danger.reasons)}"
                            )
                            self._ensure_standing(
                                sit_scan, sit_pose, stand_pose
                            )
                            return "danger"
                        if hp is not None:
                            last_hp = hp
                    if sp_stalled:
                        last_progress = now
                elif sp_stalled:
                    # Between HP polls: still check near objects on SP stall.
                    frame = self._capture_client()
                    if frame is not None:
                        danger = self._assess_danger(frame)
                        if danger.has_near_objects:
                            ctx.logger.behavior(
                                f"[SIT] danger while sitting sp={sp} "
                                f"reasons={','.join(danger.reasons)}"
                            )
                            self._ensure_standing(
                                sit_scan, sit_pose, stand_pose
                            )
                            return "danger"
                    last_progress = now

            ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)

        self._ensure_standing(sit_scan, sit_pose, stand_pose)
        ctx.logger.behavior("[SIT] stopped while sitting — stood up")
        return "stopped"

    def _measure_stand_pose(self) -> CharacterPose | None:
        """Read standing pose before the sit key; retry while unreadable."""
        for attempt in range(1, SIT_POSE_MAX_ATTEMPTS + 1):
            pose = self._measure_pose()
            if pose is not None:
                return pose
            self._ctx.logger.behavior(
                f"[SIT] stand pose unreadable attempt={attempt}/"
                f"{SIT_POSE_MAX_ATTEMPTS}"
            )
            if not self._ctx.wait_unless_stopped(SIT_POSE_SETTLE_S):
                return None
        return None

    def _ensure_sitting(
        self,
        sit_scan: int,
        stand_pose: CharacterPose,
    ) -> CharacterPose | None:
        """Press sit until center pose is shorter than stand by the min drop."""
        for attempt in range(1, SIT_POSE_MAX_ATTEMPTS + 1):
            self._input.teleport_key(sit_scan)
            if not self._ctx.wait_unless_stopped(SIT_POSE_SETTLE_S):
                return None
            current = self._measure_pose()
            if current is None:
                self._ctx.logger.behavior(
                    f"[SIT] sit pose unreadable attempt={attempt}/"
                    f"{SIT_POSE_MAX_ATTEMPTS}"
                )
                continue
            drop = stand_pose.body_height - current.body_height
            if drop >= SIT_POSE_MIN_HEIGHT_DROP:
                return current
            
            # If height increased significantly, we were already sitting and just stood up!
            # Press sit again to actually sit down, and update stand_pose to this new standing height.
            if drop <= -SIT_POSE_MIN_HEIGHT_DROP:
                self._ctx.logger.behavior(
                    f"[SIT] pose inverted! Height increased by {-drop}px. "
                    "We were already sitting. Updating stand pose and retrying."
                )
                stand_pose = current
                continue

            self._ctx.logger.behavior(
                f"[SIT] sit not confirmed attempt={attempt}/"
                f"{SIT_POSE_MAX_ATTEMPTS} stand_h={stand_pose.body_height} "
                f"current_h={current.body_height} drop={drop} "
                f"need>={SIT_POSE_MIN_HEIGHT_DROP} — retry"
            )
        return None

    def _ensure_standing(
        self,
        sit_scan: int,
        sit_pose: CharacterPose,
        stand_pose: CharacterPose,
    ) -> bool:
        """Press stand until center pose is no longer seated. Returns success."""
        mid = (sit_pose.body_height + stand_pose.body_height) / 2.0
        for attempt in range(1, SIT_POSE_MAX_ATTEMPTS + 1):
            current = self._measure_pose()
            if current is not None and current.body_height >= mid:
                return True
            self._input.teleport_key(sit_scan)
            if not self._ctx.wait_unless_stopped(SIT_POSE_SETTLE_S):
                return False
            current = self._measure_pose()
            if current is None:
                self._ctx.logger.behavior(
                    f"[SIT] stand pose unreadable attempt={attempt}/"
                    f"{SIT_POSE_MAX_ATTEMPTS}"
                )
                continue
            if current.body_height >= mid:
                return True
            self._ctx.logger.behavior(
                f"[SIT] stand not confirmed attempt={attempt}/"
                f"{SIT_POSE_MAX_ATTEMPTS} current_h={current.body_height} "
                f"mid={mid:.0f} — retry"
            )
        return False
