"""Sit when SP is low; pause hunting until SP recovers.

Sit/stand is a **toggle key**. Pose OCR (falcon, animation, crop) is too
unreliable to drive retries — each wrong read caused another tap and the
character flapped sit↔stand.

RO gameplay fact: **a seated character stands up automatically the moment
it takes any damage.** That is why every damage-escape path resets
``_seated = False`` — after a hit the character really is standing, so the
next ``sit()`` correctly presses the toggle down instead of inverting a
still-seated pose. There is no "teleport preserved the seated pose" state
to reconcile after damage.

Contract
--------
* After any damage/teleport the character is standing. ``sit()`` presses
  the key **once** and marks ``_seated``.
* ``stand()`` presses **once** only while ``_seated``, then clears the flag.
* The same relocation applies when the SP feed stays unreadable (OCR layout
  lost / panel gone) — a character parked on a dead feed can neither finish
  regen nor react to damage.
* A **readable but flat** SP value is treated the same way: a seated character's
  SP changes on every regen tick, so a value that never moves for
  ``SIT_SP_FEED_BLIND_RELOCATE_S`` means regen is blocked (re-sit toggle eaten
  during a landing — character standing while the bot believes seated — or a
  weight penalty). Recovery relocates to a fresh area, which deterministically
  lands the character standing, and re-sits there.
* Recovery is bounded: after ``SIT_MAX_SPOT_RELOCATIONS`` spot failures the
  session ends and the runtime loop takes over; the sit gate is never held
  forever by an unrecoverable spot.
* Hunt stays paused until SP ≥ resume, a danger escape, or stop.
"""

from __future__ import annotations

import time
import traceback

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import (
    SIT_KEY_SETTLE_S,
    SIT_LOW_SP_RATIO,
    SIT_MAX_SPOT_RELOCATIONS,
    SIT_POST_TELEPORT_SETTLE_S,
    SIT_RESUME_SP_RATIO,
    SIT_SP_FEED_BLIND_RELOCATE_S,
    SIT_SP_POLL_INTERVAL_S,
    SIT_SP_PROGRESS_LOG_S,
    SIT_STAND_RESUME_DELAY_S,
    WORKER_POLL_INTERVAL_S,
)
from pybot.runtime.danger_detector import DangerController, DangerDetector, DangerLevel
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
        danger_controller: DangerController | None = None,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._teleport = teleport
        self._vitals = vitals
        self._danger = danger
        self._danger_controller = danger_controller or getattr(
            ctx, "danger_controller", None
        )
        self._seated = False
        # Spot failures (blind feed) per recovery session; resets at the start
        # of each session.
        self._spot_relocations = 0
        # Seated regen watchdog state: last seen SP value and monotonic-ms time
        # it last changed. A readable value frozen for a full relocation window
        # means regeneration is blocked (mis-seated / weight penalty).
        self._sp_last_value: int | None = None
        self._sp_flat_since_ms: int | None = None
        # Regen progress telemetry timestamps (monotonic ms).
        self._regen_started_ms: int = 0
        self._last_regen_log_ms: int = 0
        ctx.register_sit_cleanup(self._retry_cleanup_stand)

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
            accepted = bool(self._input.toggle_key(sit_scan))
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
        # A leftover critical request (latched before this session claimed the
        # gate) is owned by the seated session, not by an in-flight escape —
        # the recovery loop consumes it before acting. Only an actual in-flight
        # escape may hold input; yield to it without pressing the toggle.
        if ctx.danger_escape_active.is_set():
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
        try:
            return bool(self._input.cleanup_toggle_key(sit_scan))
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
        """Return the current SP ratio, or ``None`` when SP is unavailable."""
        sp, sp_max = self._vitals.sp_pair()
        if sp is None or sp_max is None or sp_max <= 0:
            return None
        return sp / sp_max

    def _sp_observed_ms(self) -> int:
        """Last SP observation clock in monotonic ms (0 for bare fakes)."""
        try:
            return int(getattr(self._vitals, "observed_ms", 0) or 0)
        except (AttributeError, TypeError):
            return 0

    def _sp_recovered(self) -> bool:
        ratio = self._sp_ratio()
        return ratio is not None and ratio >= SIT_RESUME_SP_RATIO

    def _wait_post_teleport_settle(self) -> None:
        """Let the client finish the landing before the sit toggle is sent.

        The sit key pressed during the teleport landing transition can be
        eaten (character stays standing while the bot believes it is seated).
        A short margin after the teleport settle makes the single sit press
        land on a settled character so the logical seated state matches the
        game.
        """
        if not self._ctx.wait_unless_stopped(SIT_POST_TELEPORT_SETTLE_S):
            self._ctx.logger.behavior(
                "[SIT] post-teleport settle interrupted"
            )

    def _sit_danger_detected(self) -> bool:
        """Read the shared danger facts for the seated recovery owner."""
        return self._danger.danger_level() is not DangerLevel.SAFE

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
            return False
        if self._danger_controller is not None:
            return bool(
                self._danger_controller.escape(
                    seated=True,
                    reason=reason,
                )
            )
        if not self._ctx.begin_danger_transition(allow_sitting=True):
            return False
        try:
            escaped = bool(
                self._teleport.danger_teleport(
                    reason=reason,
                    prefer_safe_key=True,
                )
            )
            return escaped
        except Exception:
            self._ctx.logger.behavior(
                f"[SIT] urgent danger teleport failed:\n{traceback.format_exc()}"
            )
            return False
        finally:
            self._ctx.end_danger_transition()

    def process_pending(self) -> bool:
        """Advance one sit/danger decision synchronously.

        Long recovery remains one owned action sequence rather than a second
        control thread. Observation workers only publish vitals/danger state.
        """
        ctx = self._ctx
        # The worker can be called directly by lightweight integrations even
        # when runtime composition would normally omit it. A disabled feature
        # must not inspect SP, claim the sit gate, or enter recovery.
        if (
            not bool(getattr(ctx.config, "sit_on_low_sp", False))
            or not str(getattr(ctx.config, "sit_on_low_sp_button", "") or "").strip()
            or int(getattr(ctx.config, "sit_on_low_sp_scan_code", 0) or 0) <= 0
        ):
            return False
        if ctx.is_stopped() or not ctx.should_run_workers():
            return False
        if ctx.danger_escape_active.is_set():
            return False
        ratio = self._sp_ratio()
        if ratio is not None and ratio < SIT_LOW_SP_RATIO:
            self._recover_sp(ratio, reason="low_sp")
            return True
        # Danger is a fact owned by the shared detector; seated recovery
        # handles it synchronously rather than latching a request event.
        if self._danger.danger_level() is not DangerLevel.SAFE:
            return False
        return False


    def _recover_sp(
        self,
        low_ratio: float,
        *,
        reason: str = "low_sp",
    ) -> None:
        """Break hunt, move safe, sit/recover, then start a fresh hunt."""
        ctx = self._ctx
        if (
            not bool(getattr(ctx.config, "sit_on_low_sp", False))
            or not str(getattr(ctx.config, "sit_on_low_sp_button", "") or "").strip()
            or int(getattr(ctx.config, "sit_on_low_sp_scan_code", 0) or 0) <= 0
        ):
            return
        sit_scan = ctx.config.sit_on_low_sp_scan_code
        if reason == "danger":
            # Do not consume the request until the session is ours. If another
            # session owns the input boundary, the outer loop retries later.
            if not ctx.try_begin_sit_ops():
                return
        elif not ctx.begin_sit_ops():
            return
        escape_first = reason == "danger"
        teleported_for_session = False
        self._spot_relocations = 0
        try:
            ctx.logger.behavior(
                f"[SIT] {reason} session ratio={low_ratio:.1%} — hunt paused until "
                f"SP>={SIT_RESUME_SP_RATIO:.0%}"
            )
            while not ctx.is_stopped():
                # A critical request can only be latched before this session
                # claimed the sit gate: DangerDetector never queues critical
                # while a true SP sit owns ``sitting_event``, and the critical
                # gameplay owner deliberately yields to this session. That leftover
                # belongs to the seated owner, not to an in-flight escape.
                # Consume it and, if the damage is still live, escape with
                # the safe key. Never wait for a "critical handoff" — the
                # gameplay owner cannot preempt this session, so waiting
                # would park the gameplay thread forever.
                if self._danger.danger_level() is DangerLevel.CRITICAL:
                    escape_first = True
                    continue

                # A seated danger retry and a fresh hit preserved during the
                # preceding teleport settle belong to this recovery owner.
                # Inspect them before _ok_to_act(), which intentionally yields
                # to critical danger for ordinary hunting sessions.
                if self._seated and escape_first:
                    if self._urgent_escape(reason="sit_danger"):
                        # RO stands a seated character up automatically when
                        # it takes damage, so after this damage escape the
                        # character is genuinely standing. Reset the flag or
                        # the next sit() would invert the toggle.
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

                if teleported_for_session and not escape_first and self._sit_danger_detected():
                    escape_first = True
                    continue

                if not self._ok_to_act():
                    # Pending critical/escape — park, do not end recovery.
                    ctx.stop_event.wait(WORKER_POLL_INTERVAL_S)
                    continue

                # A normal sit-toggle interruption must be cleaned up by
                # standing before retrying. A damage escape is different: it
                # must retry teleport directly while still seated, never press
                # the toggle just to escape.
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
                # Preserve the fresh detector fact until the seated owner
                # handles it; no request event is latched or replayed.
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

                if outcome in ("feed_lost", "regen_stalled"):
                    # The SP feed stayed unreadable (or readable but flat — no
                    # regen ticks while seated). The escape already moved to a
                    # fresh area, so sit again there. A spot failure is not a
                    # danger escape: repeated failures mean recovery cannot
                    # complete in any spot, so the session is bounded and ends
                    # cleanly — never holding the sit gate forever on an
                    # unrecoverable spot.
                    self._spot_relocations += 1
                    if self._spot_relocations > SIT_MAX_SPOT_RELOCATIONS:
                        self._seated = False
                        ctx.logger.behavior(
                            "[SIT] sit spot failed "
                            f"{self._spot_relocations} times — ending "
                            "recovery session"
                        )
                        break
                    escape_first = False
                    teleported_for_session = True
                    reason = "low_sp"
                    self._wait_post_teleport_settle()
                    ctx.logger.behavior(
                        "[SIT] sit spot failed — relocating "
                        "(hunt stays paused)"
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
        """Release the sit session after recovery and start a fresh hunt.

        A critical request can only be latched before this session claimed
        the sit gate (the gameplay owner deliberately yields to a true SP
        sit). It belongs to the seated owner, so consume it here — leaving it
        set would block ``should_run_combat``/``should_run_timers`` forever
        because no escape is in flight to clear it. Standing then releases
        the gate and begins the next hunt generation.
        """
        ctx = self._ctx
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
        """Sit once, wait for fresh SP/damage observations, then stand once.

        The sit key is a toggle and there is no reliable seated-pose signal.
        Once the accepted sit key marks the worker seated, low but unchanged
        SP is not evidence that the character is standing: ``changed_ms`` is
        a historical value-change clock, not a sit-entry clock. Never press a
        corrective toggle from that ambiguous state. Only a recovered SP
        threshold, danger, stop, an unreadable/stale feed, or a readable SP
        frozen for a full relocation window (regen blocked) may end this
        wait. The flat-SP end case relocates instead of toggling: the escape
        teleport lands the character standing, so the re-sit re-asserts the
        pose deterministically.
        """
        ctx = self._ctx
        # Damage during teleport/idle clearing means the location is no longer
        # safe. The queued request is consumed here and the caller finds a new
        # location before trying to sit again.
        if self._sit_danger_detected():
            # This seated recovery owns the damage event. Clear a mirrored
            # critical hunting request that may have been raised in the small
            # race immediately before the sit gate was acquired, otherwise the
            # separate danger controller could duplicate the escape later.
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

        feed_unreadable_since: float | None = None
        self._regen_started_ms = int(time.monotonic() * 1000)
        self._last_regen_log_ms = self._regen_started_ms
        # The value change clock is local to this seated interval: the previous
        # session's last value is not evidence that this sit is flat.
        self._sp_last_value = None
        self._sp_flat_since_ms = None

        while not ctx.is_stopped():
            # The sit worker owns damage observed during its seated recovery.
            # Escape here on the gameplay thread; do not wait for the
            # separate danger controller (which no longer queues while a
            # true SP sit owns sitting_event).
            if self._sit_danger_detected():
                ctx.logger.behavior(
                    "[SIT] danger — urgently escaping without stand toggle"
                )
                # Damage is an emergency escape, not normal SP recovery. Do
                # not press the sit/stand toggle here: teleport immediately.
                # RO stands a seated character up automatically when it takes
                # damage, so after this escape the character is standing —
                # reset ``_seated`` accordingly.
                if not self._urgent_escape(reason="sit_danger"):
                    return "danger_escape_failed"
                self._seated = False
                return "danger_escaped"

            if not self._ok_to_act():
                return None

            # One atomic sample per iteration: the threshold, blind-feed, and
            # regen-watchdog decisions must agree on the same SP value.
            sp, sp_max = self._vitals.sp_pair()
            ratio = (
                sp / sp_max
                if sp is not None and sp_max is not None and sp_max > 0
                else None
            )
            if ratio is not None and ratio >= SIT_RESUME_SP_RATIO:
                ctx.logger.behavior(
                    f"[SIT] SP threshold met ratio={ratio:.1%} — standing"
                )
                if not self.stand(sit_scan):
                    return None
                if not ctx.wait_unless_stopped(SIT_STAND_RESUME_DELAY_S):
                    return None
                # Do not release the recovery session until the post-stand
                # client frame is settled. The gameplay owner then starts the
                # new generation; discovery/tracking cannot overlap the stand
                # animation or the first buff/timer input.
                if not self._sp_recovered():
                    ctx.logger.behavior(
                        "[SIT] SP dropped below resume after stand — not done"
                    )
                    return None
                return "recovered"

            now = time.monotonic()
            # SP unreadable (panel missing → feed clears the value) or the
            # observation stream stalled (digits unreadable / wedged read).
            # Recovery can neither confirm regeneration nor react to damage,
            # so waiting here parks the runtime forever — the "OCR layout
            # disappeared, bot doing nothing" state. Ride out a short spell
            # while the feed self-heals, then relocate to a fresh area.
            if ratio is None or self._sp_feed_stale(SIT_SP_FEED_BLIND_RELOCATE_S):
                if feed_unreadable_since is None:
                    feed_unreadable_since = now
                elif now - feed_unreadable_since >= SIT_SP_FEED_BLIND_RELOCATE_S:
                    ctx.logger.behavior(
                        "[SIT] SP feed unreadable too long — relocating "
                        "(hunt stays paused)"
                    )
                    if not self._urgent_escape(reason="sit_feed_lost"):
                        return "danger_escape_failed"
                    self._seated = False
                    return "feed_lost"
                ctx.stop_event.wait(WORKER_POLL_INTERVAL_S)
                continue
            feed_unreadable_since = None

            # Regen watchdog + progress visibility. While the feed is alive a
            # seated character's SP changes on every regen tick, so a value
            # frozen for the whole relocation window means regen is blocked —
            # most often the character is not actually seated (the re-sit
            # toggle was eaten during a landing transition) or a weight
            # penalty has stopped regen. Waiting longer only parks the
            # runtime, so relocate to a fresh area (teleport lands the
            # character standing) and re-sit there.
            # The flat clock is intentionally strict about *movement*: any
            # value change — including a single OCR digit jitter — restarts it,
            # so it never relocates a genuinely regenerating character. The
            # trade-off is a jittery feed could keep the clock fresh and mask a
            # truly stuck value; the blind-feed window still covers a feed that
            # stops publishing altogether.
            now_ms = int(now * 1000)
            if sp != self._sp_last_value:
                self._sp_last_value = sp
                self._sp_flat_since_ms = now_ms
            if (
                self._sp_flat_since_ms is not None
                and sp is not None
                and now_ms - self._sp_flat_since_ms
                >= int(SIT_SP_FEED_BLIND_RELOCATE_S * 1000)
            ):
                weight = self._vitals.weight_pair()
                weight_text = (
                    f"{weight[0]}/{weight[1]}"
                    if weight[0] is not None
                    else "n/a"
                )
                ctx.logger.behavior(
                    f"[SIT] SP flat while seated sp={sp}/{sp_max} "
                    f"weight={weight_text} — relocating (hunt stays paused)"
                )
                if not self._urgent_escape(reason="sit_regen_stalled"):
                    return "danger_escape_failed"
                self._seated = False
                return "regen_stalled"
            if now_ms - self._last_regen_log_ms >= int(
                SIT_SP_PROGRESS_LOG_S * 1000
            ):
                self._last_regen_log_ms = now_ms
                ratio_text = "—"
                if sp is not None and sp_max:
                    ratio_text = f"{sp / sp_max:.0%}"
                ctx.logger.behavior(
                    f"[SIT] regen sp={sp}/{sp_max} ratio={ratio_text} "
                    f"elapsed={(now_ms - self._regen_started_ms) // 1000}s"
                )

            # Poll SP on the sit cadence, but wake often enough that seated
            # danger (owned here) remains reactive without a second control
            # thread pressing teleport keys.
            ctx.stop_event.wait(WORKER_POLL_INTERVAL_S)
        return None

    def _sp_feed_stale(self, max_age_s: float) -> bool:
        """True when no SP observation has arrived for ``max_age_s``.

        Note: when the panel goes missing the feed clears SP via
        ``publish_sp(None, None)``, which refreshes ``observed_ms`` — a
        missing panel therefore looks *fresh* here, and the caller's
        ``ratio is None`` check is what catches that case. This helper only
        catches a stalled publish stream (digits unreadable / wedged read)
        where the last value persists while observations stop — recovery
        can neither confirm regeneration nor react to damage either way.
        """
        observed_ms = self._sp_observed_ms()
        if observed_ms <= 0:
            # Bare fakes / never-published feeds have no clock; never treat
            # them as a stalled real feed.
            return False
        return (time.monotonic() * 1000) - observed_ms > int(max_age_s * 1000)
