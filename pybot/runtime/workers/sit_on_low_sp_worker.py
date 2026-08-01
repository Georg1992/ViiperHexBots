"""Sit when SP is low; pause hunting until SP recovers.

Sit/stand is a **toggle key**. Pose OCR (falcon, animation, crop) is too
unreliable to drive retries — each wrong read caused another tap and the
character flapped sit↔stand.

Contract
--------
* After area-clear the character is standing (teleport). ``sit()`` presses
  the key **once** and marks ``_seated``.
* ``stand()`` presses **once** only while ``_seated``, then clears the flag.
* No pose reads. No second tap. Hunt stays paused until SP ≥ resume or stop.
"""

from __future__ import annotations

import traceback

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import (
    SIT_KEY_SETTLE_S,
    SIT_LOW_SP_RATIO,
    SIT_RESUME_SP_RATIO,
    SIT_SP_POLL_INTERVAL_S,
    SIT_STAND_RESUME_DELAY_S,
)
from pybot.runtime.danger_detector import DangerDetector
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.teleport import TeleportController
from pybot.runtime.workers.worker_contexts import SitOnLowSpWorkerContext


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

    def _tap(self, sit_scan: int, *, why: str) -> bool:
        self._ctx.logger.behavior(f"[SIT] tap sit-key reason={why}")
        self._input.key_tap(sit_scan)
        return self._ctx.wait_unless_stopped(SIT_KEY_SETTLE_S)

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
        """Press sit once. After teleport-clear we are standing — no pose check."""
        if self._seated:
            return True
        if not self._ok_to_act():
            return False
        if not self._tap(sit_scan, why="enter_sit"):
            return False
        self._seated = True
        self._ctx.logger.behavior("[SIT] seated")
        return True

    def stand(self, sit_scan: int) -> bool:
        """Press stand once if seated. No pose verify — never a second toggle."""
        if not self._seated:
            return True
        if not self._ok_to_act():
            return False
        # Clear flag before settle wait so a retry/finally cannot tap again.
        self._seated = False
        ok = self._tap(sit_scan, why="leave_sit")
        if ok:
            self._ctx.logger.behavior("[SIT] standing")
        return ok

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
                if ctx.danger_sit_requested.is_set():
                    # Keep the request queued until begin_sit_ops() acquires
                    # ownership. Storage/heal may be holding the session lock;
                    # consuming first would lose the escape request.
                    ratio = self._sp_ratio()
                    self._recover_sp(
                        ratio if ratio is not None else 1.0,
                        reason="danger",
                        consume_danger_request=True,
                    )
                    # Storage/heal may still own the session. Avoid a hot
                    # retry loop while the request remains queued.
                    if ctx.danger_sit_requested.is_set():
                        ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                    continue
                ratio = self._sp_ratio()
                if ratio is None:
                    ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                    continue
                if ratio < SIT_LOW_SP_RATIO:
                    self._recover_sp(ratio, reason="low_sp")
                else:
                    ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
            except Exception:
                ctx.logger.behavior(f"[SIT] tick error:\n{traceback.format_exc()}")

    def _recover_sp(
        self,
        low_ratio: float,
        *,
        reason: str = "low_sp",
        consume_danger_request: bool = False,
    ) -> None:
        """Break hunt, move safe, sit/recover, then start a fresh hunt."""
        ctx = self._ctx
        sit_scan = ctx.config.sit_on_low_sp_scan_code
        self._seated = False
        if consume_danger_request:
            # Danger requests are retried by the outer loop if storage/heal
            # currently owns the session; never block while holding the queue.
            if not ctx.try_begin_sit_ops():
                return
            # The request is cleared only after sit ownership is held.
            ctx.pop_danger_sit_request()
        elif not ctx.begin_sit_ops():
            return
        try:
            ctx.logger.behavior(
                f"[SIT] {reason} session ratio={low_ratio:.1%} — hunt paused until "
                f"SP>={SIT_RESUME_SP_RATIO:.0%}"
            )
            while not ctx.is_stopped():
                if not self._ok_to_act():
                    break

                if not self._teleport.teleport_to_safe_place(log_tag="SIT"):
                    ctx.logger.behavior(
                        "[SIT] area clear stopped — retry (hunt stays paused)"
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
            self.stand(sit_scan)  # no-op if already stood
            ctx.end_sit_ops()
            ctx.discovery_wake.set()

    def _sit_until_done(self, sit_scan: int) -> str | None:
        """Sit once → wait SP/damage → stand once."""
        ctx = self._ctx
        # Damage during teleport/idle clearing means the location is no longer
        # safe. Preserve the event and let _recover_sp clear a new location
        # before trying to sit again.
        if self._danger.pop_damage_detected():
            ctx.logger.behavior(
                "[SIT] damage observed during area clear — finding a new spot"
            )
            return "interrupted"

        if not self.sit(sit_scan):
            ctx.logger.behavior("[SIT] sit interrupted — will retry")
            return None
        ctx.logger.behavior("[SIT] waiting for regen")

        while not ctx.is_stopped():
            if not self._ok_to_act():
                return None

            if self._danger.pop_damage_detected():
                ctx.logger.behavior("[SIT] damage — standing before new spot")
                if not self.stand(sit_scan):
                    return None
                return "interrupted"

            ratio = self._sp_ratio()
            if ratio is not None and ratio >= SIT_RESUME_SP_RATIO:
                ctx.logger.behavior(
                    f"[SIT] SP threshold met ratio={ratio:.1%} — standing"
                )
                if not self.stand(sit_scan):
                    return None
                if not ctx.wait_unless_stopped(SIT_STAND_RESUME_DELAY_S):
                    return None
                if not self._sp_recovered():
                    ctx.logger.behavior(
                        "[SIT] SP dropped below resume after stand — not done"
                    )
                    return None
                return "recovered"

            ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
        return None
