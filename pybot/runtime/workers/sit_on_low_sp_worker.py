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
        """Consume a danger request raised by an observed HP drop.

        Nearby mobs alone are not an attack signal: they can be present while
        the character is safely sitting. The danger detector queues this event
        only after seeing HP decrease, so the sit worker remains the sole owner
        of the resulting escape sequence.
        """
        return self._ctx.pop_danger_sit_request()

    def _urgent_escape(self, *, reason: str) -> bool:
        """Escape danger, retaining the request until teleport succeeds."""
        # Never send emergency input after an explicit stop. A pause keeps the
        # request pending so resume can retry it before hunting continues.
        if self._ctx.is_stopped():
            return False
        if self._ctx.pause_event.is_set():
            self._ctx.request_danger_sit()
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
                "[SIT] urgent danger teleport unavailable — retrying before "
                "safe-place search"
            )
        # A successful teleport resets the old damage sample, but it must not
        # clear a request raised by a newer HP drop during teleport settle.
        # The sit session consumed the request it owns before calling here.
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
                    # The request remains set until this worker owns the sit
                    # session. Storage/heal may be using the input boundary.
                    ratio = self._sp_ratio()
                    self._recover_sp(
                        ratio if ratio is not None else 1.0,
                        reason="danger",
                    )
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
    ) -> None:
        """Break hunt, move safe, sit/recover, then start a fresh hunt."""
        ctx = self._ctx
        sit_scan = ctx.config.sit_on_low_sp_scan_code
        if reason == "danger":
            # Do not consume the request until the session is ours. If another
            # session owns the input boundary, the outer loop retries later.
            if not ctx.try_begin_sit_ops():
                return
            ctx.pop_danger_sit_request()
        elif not ctx.begin_sit_ops():
            return
        escape_first = reason == "danger"
        teleported_for_session = False
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

                # A pause/input interruption may happen after SP has already
                # recovered. Do not start another teleport/sit cycle; the
                # character is standing here, so the hunt can resume directly.
                if not escape_first and self._sp_recovered():
                    ratio = self._sp_ratio()
                    ctx.logger.behavior(
                        f"[SIT] SP recovered during retry ratio={ratio:.1%} "
                        "— resuming hunt"
                    )
                    break

                if escape_first:
                    # Damage interrupted hunting or a previous sit attempt.
                    # Escape first, then use the same quiet-area search as low
                    # SP recovery. A failed escape leaves the request handled;
                    # the safe-place loop still retries while hunting is held.
                    if not self._urgent_escape(reason="sit_danger_request"):
                        ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                        continue
                    escape_first = False
                    teleported_for_session = True

                # A fresh damage event raised while the urgent escape was in
                # flight takes priority over normal quiet-area searching.
                if self._sit_danger_detected():
                    # Consume fresh damage before retrying. Leaving this event
                    # set replays one request forever after an urgent escape,
                    # especially when damage arrives during teleport settle.
                    escape_first = True
                    continue

                if not teleported_for_session:
                    if not self._teleport.teleport_once_for_sit(log_tag="SIT"):
                        ctx.logger.behavior(
                            "[SIT] teleport stopped — retry (hunt stays paused)"
                        )
                        ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                        continue
                    teleported_for_session = True

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
                    # _sit_until_done already stood and performed the urgent
                    # teleport. Sit directly in that new area; do not send a
                    # second normal teleport during the same recovery session.
                    escape_first = False
                    # The interrupted result is only returned after an urgent
                    # escape succeeds; that escape is this session's one
                    # teleport.
                    teleported_for_session = True
                    reason = "low_sp"
                    ctx.logger.behavior(
                        "[SIT] interrupted — new sit spot (hunt stays paused)"
                    )
                    continue

                if outcome == "danger_escape_failed":
                    # Stay in the danger phase until the urgent escape succeeds;
                    # never search for a sit spot from the unsafe area.
                    reason = "danger"
                    escape_first = True
                    ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                    continue

                if self._sit_danger_detected():
                    # A failed stand re-queued danger. Consume it only after
                    # the stand path has completed, then escape before any
                    # quiet-area search from the old location.
                    reason = "danger"
                    escape_first = True
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
        # safe. The queued request is consumed here and the caller finds a new
        # location before trying to sit again.
        if self._sit_danger_detected():
            ctx.logger.behavior(
                "[SIT] danger observed before sitting — urgent escape "
                "and finding a new spot"
            )
            if not self._urgent_escape(reason="sit_spot_danger"):
                return "danger_escape_failed"
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
                    # _sit_danger_detected() may have consumed the only queued
                    # event. Keep danger pending until the seated toggle is
                    # successfully undone and escape can run.
                    ctx.request_danger_sit()
                    return None
                if not self._urgent_escape(reason="sit_danger"):
                    return "danger_escape_failed"
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
