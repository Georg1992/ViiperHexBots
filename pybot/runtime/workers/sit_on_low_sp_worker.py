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
        register_cleanup = getattr(ctx, "register_sit_cleanup", None)
        if callable(register_cleanup):
            register_cleanup(self._retry_cleanup_stand)

    def _tap(self, sit_scan: int, *, why: str) -> bool:
        """Send one toggle and report whether the input was accepted.

        The settle wait may be interrupted by Stop/Pause after the key has
        already been sent. Callers must therefore use the key result—not the
        wait result—to update the logical seated state.
        """
        self._ctx.logger.behavior(f"[SIT] tap sit-key reason={why}")
        try:
            toggle = getattr(self._input, "toggle_key", None)
            if callable(toggle):
                accepted = bool(toggle(sit_scan))
            else:
                # Compatibility for narrow legacy test/custom backends.
                accepted = bool(self._input.key_tap(sit_scan, after_s=0.0))
        except Exception:
            self._ctx.logger.behavior(
                f"[SIT] sit-key input failed reason={why}:\n"
                f"{traceback.format_exc()}"
            )
            return False
        if not accepted:
            self._ctx.logger.behavior(
                f"[SIT] sit-key input rejected reason={why}"
            )
            return False
        if not self._ctx.wait_unless_stopped(SIT_KEY_SETTLE_S):
            self._ctx.logger.behavior(
                f"[SIT] sit-key settle interrupted reason={why}"
            )
        # The toggle was accepted even if Stop/Pause interrupted the settle.
        return True

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
        # Keep the flag set until the stand key is accepted. If input is
        # rejected, cleanup/retry must know the character may still be seated.
        ok = self._tap(sit_scan, why="leave_sit")
        if not ok:
            return False
        self._seated = False
        self._ctx.logger.behavior("[SIT] standing")
        return True

    def _cleanup_stand(self, sit_scan: int) -> bool:
        """Undo an accepted sit toggle during shutdown.

        Normal input is cancelled before worker joins so long macros unwind.
        A seated character is different: leaving it seated would make the next
        runtime's first toggle invert the state. The dedicated backend method
        is allowed to emit only this final key pair after cancellation.
        """
        cleanup = getattr(self._input, "cleanup_toggle_key", None)
        if not callable(cleanup):
            return False
        try:
            return bool(cleanup(sit_scan))
        except Exception:
            self._ctx.logger.behavior(
                f"[SIT] shutdown stand failed:\n{traceback.format_exc()}"
            )
            return False

    def _retry_cleanup_stand(self) -> bool:
        """Retry the one unresolved stand toggle during runtime shutdown."""
        if not self._seated:
            return True
        sit_scan = self._ctx.config.sit_on_low_sp_scan_code
        if not self._cleanup_stand(sit_scan):
            return False
        self._seated = False
        self._ctx.logger.behavior("[SIT] shutdown stand accepted on retry")
        return True

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

    def _sit_danger_detected(self) -> bool:
        """Consume damage or detect any active nearby threat at the sit spot."""
        if self._danger.pop_damage_detected():
            return True
        nearby = getattr(self._danger, "has_nearby_threat", None)
        return callable(nearby) and bool(nearby())

    def _urgent_escape(self, *, reason: str) -> bool:
        """Immediately escape a damaged or otherwise dangerous sit spot."""
        # Never send emergency input after an explicit stop/pause. A pause may
        # arrive between damage polling and this call, so check at the final
        # boundary as well as in the normal action helpers.
        if self._ctx.is_stopped() or self._ctx.pause_event.is_set():
            return False
        try:
            escaped = self._teleport.danger_teleport(reason=reason)
        except Exception:
            self._ctx.logger.behavior(
                f"[SIT] urgent danger teleport failed:\n{traceback.format_exc()}"
            )
            return False
        if not escaped:
            self._ctx.logger.behavior(
                "[SIT] urgent danger teleport unavailable — "
                "continuing with safe-place retry"
            )
        return bool(escaped)

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

                # A previous interrupted attempt may have sent the sit toggle
                # but failed its settle wait. Never teleport or press sit again
                # while that logical seated state is still owned by this
                # worker; stand first, retrying while the session is held.
                if self._seated:
                    if not self.stand(sit_scan):
                        ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                    continue

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
            # Do not release the sit gate while we still believe the character
            # is seated. Retry transient input rejection while running; Pause
            # intentionally waits for resume. Stop uses the shutdown-only
            # toggle path because normal input has already been cancelled.
            shutdown_cleanup_attempts = 0
            while self._seated:
                if ctx.is_stopped():
                    shutdown_cleanup_attempts += 1
                    if self._cleanup_stand(sit_scan):
                        self._seated = False
                        ctx.logger.behavior("[SIT] shutdown stand accepted")
                        break
                    if shutdown_cleanup_attempts >= 3:
                        # Never let an unavailable input backend deadlock the
                        # entire runtime. The character state is explicitly
                        # unresolved; the failure is visible in the log rather
                        # than being mistaken for a successful stand.
                        ctx.logger.behavior(
                            "[SIT] shutdown stand could not be confirmed "
                            "after 3 attempts"
                        )
                        mark_unresolved = getattr(
                            ctx, "mark_sit_cleanup_unresolved", None
                        )
                        if callable(mark_unresolved):
                            mark_unresolved()
                        break
                    continue
                if ctx.pause_event.is_set():
                    # Keep the sit gate held while the character is still
                    # seated. Once the user resumes, stand cleanly before
                    # releasing the gate; otherwise hunting could resume
                    # while the game character remains seated.
                    ctx.wait_while_user_paused(SIT_SP_POLL_INTERVAL_S)
                    continue
                if self.stand(sit_scan):
                    break
                ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
            ctx.end_sit_ops()
            ctx.discovery_wake.set()

    def _sit_until_done(self, sit_scan: int) -> str | None:
        """Sit once → wait SP/damage → stand once."""
        ctx = self._ctx
        # Damage during teleport/idle clearing means the location is no longer
        # safe. Preserve the event and let _recover_sp clear a new location
        # before trying to sit again.
        if self._sit_danger_detected():
            ctx.logger.behavior(
                "[SIT] danger observed before sitting — urgent escape "
                "and finding a new spot"
            )
            self._urgent_escape(reason="sit_spot_danger")
            return "interrupted"

        if not self.sit(sit_scan):
            ctx.logger.behavior("[SIT] sit interrupted — will retry")
            return None
        ctx.logger.behavior("[SIT] waiting for regen")

        while not ctx.is_stopped():
            if not self._ok_to_act():
                return None

            if self._sit_danger_detected():
                ctx.logger.behavior(
                    "[SIT] danger — standing and urgently escaping before "
                    "finding a new spot"
                )
                # Damage always wins over SP recovery: leave the seated state
                # first, then use the emergency teleport path. The outer
                # recovery loop will clear/idle the new area before sitting.
                stood = self.stand(sit_scan)
                # Never teleport while the stand toggle was rejected: the
                # worker still owns a seated state and must retry standing
                # before moving to another area. An accepted toggle remains
                # authoritative even if its settle wait was interrupted.
                if not stood:
                    return None
                self._urgent_escape(reason="sit_danger")
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
