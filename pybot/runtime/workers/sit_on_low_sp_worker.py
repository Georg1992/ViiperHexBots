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
    SIT_POST_TELEPORT_SETTLE_S,
    SIT_RESUME_SP_RATIO,
    SIT_SP_POLL_INTERVAL_S,
    SIT_STAND_RESUME_DELAY_S,
)
from pybot.runtime.danger_detector import DangerDetector, DangerLevel
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
        # A critical escape owns urgent input. Never interleave a sit toggle
        # with the escape's teleport key: after the escape's teleport the
        # character is standing, so a late toggle would invert the pose.
        if self._ctx.danger_escape_active.is_set():
            self._ctx.logger.behavior(
                f"[SIT] sit-key input skipped reason={why} — "
                "critical escape in flight"
            )
            return False
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
        # A pending or in-flight critical escape owns urgent input and the
        # sit gate. Yield to it; the outer loop abandons the session so the
        # escape worker can teleport without competing for the toggle.
        if ctx.critical_danger_requested.is_set() or ctx.danger_escape_active.is_set():
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

    def _wait_post_teleport_settle(self) -> None:
        """Let the client finish the landing before the sit toggle is sent.

        The sit key pressed during the teleport landing transition can be
        eaten (character stays standing while the bot believes it is seated)
        or, on a client that preserves the seated pose through the teleport,
        inverted. A short margin after the teleport settle makes the single
        sit press land on a settled character so the logical seated state
        matches the game.
        """
        if not self._ctx.wait_unless_stopped(SIT_POST_TELEPORT_SETTLE_S):
            self._ctx.logger.behavior(
                "[SIT] post-teleport settle interrupted"
            )

    def _sit_danger_detected(self) -> bool:
        """Consume a danger request raised by an observed HP drop.

        Nearby mobs alone are not an attack signal: they can be present while
        the character is safely sitting. The danger detector queues this event
        only after seeing HP decrease, so the sit worker remains the sole owner
        of the resulting escape sequence.
        """
        return self._ctx.pop_danger_sit_request()

    def _hunting_danger_is_critical(self) -> bool | None:
        """Return whether a pending hunting damage event warrants teleport.

        The detector queues every HP drop so damage can interrupt a seated
        recovery session. While hunting, ordinary damage is intentionally
        ignored; only the detector's CRITICAL classification may escape.
        """
        try:
            return self._danger.danger_level() is DangerLevel.CRITICAL
        except Exception:
            self._ctx.logger.behavior(
                "[DANGER] critical check failed — keeping hunt in place"
            )
            return None

    def _handle_hunting_danger(self) -> None:
        """Consume hunting damage; teleport only for critical danger."""
        ctx = self._ctx
        critical = self._hunting_danger_is_critical()
        if critical is None:
            # A detector failure must fail closed: never resume combat while a
            # damage request cannot be classified safely.
            ctx.logger.behavior(
                "[DANGER] damage classification unavailable — request retained"
            )
            return
        # The independent critical-danger worker owns hunting teleports. This
        # sit worker only consumes ordinary mirrored requests; the critical
        # request remains for the escape worker.
        if not critical:
            ctx.pop_danger_sit_request()
        else:
            ctx.logger.behavior(
                "[DANGER] critical hunting damage — delegated to escape worker"
            )
        if not critical:
            # Ordinary hits must not claim the sit gate: doing so would reset
            # hunt startup/buff state on every attack.
            ctx.logger.behavior(
                "[DANGER] ordinary hunting damage — no teleport"
            )

    def _urgent_escape(self, *, reason: str) -> bool:
        """Escape danger, retaining the request until teleport succeeds.

        Recovery-session escapes use the safe teleport key (creamy / save point
        first — the same key as the sit placement). The urgent random fly wing
        can drop the character back next to mobs, so a seated recovery would
        re-escape forever instead of completing SP recovery. The independent
        critical hunting escape keeps the random wing.
        """
        # Never send emergency input after an explicit stop. A pause keeps the
        # request pending so resume can retry it before hunting continues.
        if self._ctx.is_stopped():
            return False
        if self._ctx.pause_event.is_set():
            self._ctx.request_danger_sit()
            return False
        begin_escape = getattr(self._ctx, "begin_danger_escape", None)
        escape_owned = bool(begin_escape()) if callable(begin_escape) else False
        if callable(begin_escape) and not escape_owned:
            return False
        try:
            try:
                escaped = self._teleport.danger_teleport(
                    reason=reason,
                    prefer_safe_key=True,
                )
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
            return bool(escaped)
        finally:
            if escape_owned:
                end_escape = getattr(self._ctx, "end_danger_escape", None)
                if callable(end_escape):
                    end_escape()
        # A successful teleport resets the old damage sample, but it must not
        # clear a request raised by a newer HP drop during teleport settle.
        # The sit session consumed the request it owns before calling here.
        return bool(escaped)

    def process_pending(self) -> bool:
        """Advance one sit/danger decision synchronously.

        Long recovery remains one owned action sequence rather than a second
        control thread. Observation workers only publish vitals/danger state.
        """
        ctx = self._ctx
        if ctx.is_stopped() or not ctx.should_run_workers():
            return False
        if ctx.critical_danger_requested.is_set() or ctx.danger_escape_active.is_set():
            return False
        if ctx.danger_sit_requested.is_set():
            self._handle_hunting_danger()
            return False
        ratio = self._sp_ratio()
        if ratio is not None and ratio < SIT_LOW_SP_RATIO:
            self._recover_sp(ratio, reason="low_sp")
            return True
        return False

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
                if ctx.critical_danger_requested.is_set() or ctx.danger_escape_active.is_set():
                    # A critical escape owns urgent input and the sit gate.
                    # Park until it resolves; SP recovery resumes afterwards.
                    ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                    continue
                if ctx.danger_sit_requested.is_set():
                    # Every HP drop is queued so a seated session can escape
                    # immediately. While hunting, only CRITICAL damage may
                    # teleport; ordinary hits are consumed without recovery.
                    self._handle_hunting_danger()
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
                if ctx.stop_event.wait(0.25):
                    break

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
            # The sit worker owns this danger escape, so consume the mirrored
            # critical request as well. Otherwise the independent escape
            # worker can issue a duplicate teleport after this session ends.
            pop_critical = getattr(ctx, "pop_critical_danger", None)
            if callable(pop_critical):
                pop_critical()
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

                # A normal sit-toggle interruption must be cleaned up by
                # standing before retrying. A damage escape is different: it
                # must retry teleport directly while still seated, never press
                # the toggle just to escape.
                if self._seated:
                    if escape_first:
                        if self._urgent_escape(reason="sit_danger"):
                            # The successful retry moved this same recovery
                            # session to a new landing. Keep ownership here;
                            # ending now would make the outer gameplay tick
                            # start a second low-SP session and press the sit
                            # toggle again. Reuse this landing and perform the
                            # normal single re-sit below.
                            self._seated = False
                            escape_first = False
                            teleported_for_session = True
                            self._wait_post_teleport_settle()
                            ctx.logger.behavior(
                                "[SIT] danger escape succeeded while seated — "
                                "resuming recovery in new area"
                            )
                            continue
                        ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                        continue
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
                    self._wait_post_teleport_settle()

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
                    # A normal recovery interruption uses the new area for a
                    # follow-up sit. Damage escape has its own terminal result
                    # below and must not re-enter the sit/stand cycle.
                    escape_first = False
                    teleported_for_session = True
                    reason = "low_sp"
                    self._wait_post_teleport_settle()
                    ctx.logger.behavior(
                        "[SIT] interrupted — new sit spot (hunt stays paused)"
                    )
                    continue

                if outcome == "danger_escaped":
                    # Damage while seated is an interruption, not the end of
                    # SP recovery. The emergency teleport already moved to a
                    # new area, so sit again there without another teleport.
                    escape_first = False
                    teleported_for_session = True
                    reason = "low_sp"
                    self._wait_post_teleport_settle()
                    ctx.logger.behavior(
                        "[SIT] danger escaped — sitting again for SP recovery"
                    )
                    continue

                if outcome == "danger_escape_failed":
                    # Stay in the danger phase until the urgent escape succeeds;
                    # retry directly while seated and never stand as part of
                    # this emergency path.
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
            self._finish_recovery_session(sit_scan)

    def _finish_recovery_session(self, sit_scan: int) -> None:
        """Release the sit session after recovery, yielding to a critical escape.

        A critical escape can claim ownership while the teardown is running.
        Once it does, the escape's teleport stands the character and its
        ``end_critical_escape_ops`` clears the gate it set. Never press the
        sit toggle or run ``end_sit_ops`` after that point: a late toggle
        would invert the post-teleport pose.
        """
        ctx = self._ctx
        if ctx.danger_escape_active.is_set():
            self._seated = False
            ctx.logger.behavior(
                "[SIT] recovery session preempted by critical escape"
            )
            return
        if ctx.critical_danger_requested.is_set():
            # The escape is queued but not yet claimed. Yield one poll so the
            # critical worker can take ownership; if the request is consumed
            # as stale instead, fall through to the normal teardown on the
            # next wake.
            ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
            if ctx.danger_escape_active.is_set():
                self._seated = False
                return
        # Do not release the sit gate while we still believe the character
        # is seated. Retry transient input rejection while running; Pause
        # intentionally waits for resume. Stop uses the shutdown-only
        # toggle path because normal input has already been cancelled.
        shutdown_cleanup_attempts = 0
        while self._seated:
            if ctx.danger_escape_active.is_set():
                self._seated = False
                ctx.logger.behavior(
                    "[SIT] stand aborted — critical escape owns the character"
                )
                return
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
            # This seated recovery owns the damage event. Clear a mirrored
            # critical hunting request that may have been raised in the small
            # race immediately before the sit gate was acquired, otherwise the
            # independent critical worker could duplicate the escape later.
            pop_critical = getattr(ctx, "pop_critical_danger", None)
            if callable(pop_critical):
                pop_critical()
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
                # A seated session owns any damage request; prevent a mirrored
                # critical hunting request from escaping a second time.
                pop_critical = getattr(ctx, "pop_critical_danger", None)
                if callable(pop_critical):
                    pop_critical()
                ctx.logger.behavior(
                    "[SIT] danger — urgently escaping without stand toggle"
                )
                # Damage is an emergency escape, not normal SP recovery. Do
                # not press the sit/stand toggle here: teleport immediately
                # and preserve the logical seated state until the new area is
                # settled. This avoids an unnecessary input delay and prevents
                # a second toggle from accidentally seating the character.
                if not self._urgent_escape(reason="sit_danger"):
                    return "danger_escape_failed"
                # Teleport ends this seated recovery session. Do not let the
                # recovery loop send a redundant stand or sit toggle afterward.
                self._seated = False
                return "danger_escaped"

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
