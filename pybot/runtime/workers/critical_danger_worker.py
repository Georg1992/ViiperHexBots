"""Escape hunting characters only when the danger detector reports CRITICAL."""

from __future__ import annotations

from pybot.runtime.constants import (
    CRITICAL_DANGER_POLL_INTERVAL_S,
    CRITICAL_PREEMPT_RELEASE_TIMEOUT_S,
)
from pybot.runtime.danger_detector import DangerLevel
from pybot.runtime.teleport import TeleportController


class CriticalDangerWorker:
    """Perform one urgent teleport for each critical hunting danger event.

    Critical danger is the highest-priority signal in the runtime: the escape
    claims the input boundary even while another session (sit recovery,
    storage UI, heal) is active, and it polls at the danger detector's
    cadence so a critical drop is answered within ~100ms.
    """

    def __init__(self, ctx, teleport: TeleportController) -> None:
        self._ctx = ctx
        self._teleport = teleport

    def process_pending(self) -> bool:
        """Handle one pending critical escape; return whether it succeeded."""
        ctx = self._ctx
        # Stop and focus-loss pause must not turn an interrupted teleport into
        # a retry loop or admit emergency input after shutdown begins. Leave the
        # request queued for a later resume when paused.
        if ctx.is_stopped() or ctx.pause_event.is_set():
            return False
        # Sitting and storage do NOT block the escape: critical danger always
        # overrides any active session. The claim preempts the session owner
        # (which abandons on seeing ``danger_escape_active``) and the bounded
        # release wait below lets storage close its UI panels before the key
        # is pressed.
        if not ctx.critical_danger_requested.is_set():
            return False
        # The character may have recovered (sit regeneration, a heal, or the
        # damage simply aged out) since the drop was observed. Never fire a
        # random-wing teleport on a stale request — consume it and let the
        # hunt continue; the safe-heal path tops HP back up. The mirrored
        # requests are popped between two level reads: a fresh critical hit
        # queued in that window is restored below instead of being lost.
        danger = getattr(ctx, "danger_detector", None)
        if danger is not None:
            level = getattr(danger, "danger_level", None)
            if callable(level):
                try:
                    if level() is not DangerLevel.CRITICAL:
                        ctx.pop_critical_danger()
                        ctx.pop_danger_sit_request()
                        if level() is DangerLevel.CRITICAL:
                            # A fresh critical hit landed between the reads.
                            # Restore the request so the next poll escapes.
                            ctx.request_critical_danger()
                            return False
                        behavior = getattr(ctx.logger, "behavior", None)
                        if callable(behavior):
                            behavior(
                                "[DANGER] stale critical request consumed — "
                                "character no longer in critical danger"
                            )
                        return True
                except Exception:
                    # run() has no exception guard; a detector failure must
                    # not kill the worker thread. Fall through to the normal
                    # claim/escape path with the request still queued.
                    ctx.request_critical_danger()
        claim_critical = getattr(ctx, "try_begin_critical_escape_ops", None)
        if callable(claim_critical):
            # Always override: critical danger preempts sit/storage/heal.
            if not claim_critical(override=True):
                return False
            # A preempted storage session must close its panels before the
            # teleport key is pressed or the wing is wasted. The wait is
            # bounded; on timeout the escape presses anyway rather than
            # leaving the character in critical danger.
            wait_release = getattr(ctx, "wait_for_preempted_session_release", None)
            if callable(wait_release):
                wait_release(CRITICAL_PREEMPT_RELEASE_TIMEOUT_S)
        elif not ctx.try_begin_sit_ops():
            return False

        # A preempted sit session keeps the safe-key escape (creamy / save
        # point first, the same key the recovery session would use): the
        # character lands somewhere it can sit and finish SP recovery. The
        # random fly wing can drop it back next to mobs and cause a repeat
        # escape->sit loop. Standing hunting escapes keep the urgent wing.
        prefer_safe_key = False
        preempted = getattr(ctx, "preempted_sessions", None)
        if callable(preempted):
            try:
                prefer_safe_key = bool(preempted()[0])
            except Exception:
                prefer_safe_key = False
        try:
            # Re-check after claiming the gate because focus can be lost between
            # the initial check and the ownership transition.
            if ctx.pause_event.is_set():
                return False
            # Compatibility contexts may not expose the atomic critical claim.
            # Claim the explicit escape phase before consuming the event.
            begin_escape = getattr(ctx, "begin_danger_escape", None)
            if not callable(claim_critical) and callable(begin_escape):
                if not begin_escape():
                    return False
            # Consume both mirrored requests before teleport. A seated session
            # cannot claim this path because it already holds the sit gate.
            if not ctx.pop_critical_danger():
                if not callable(claim_critical) and callable(begin_escape):
                    ctx.end_danger_escape()
                return False
            ctx.pop_danger_sit_request()
            # Pause may arrive while consuming the mirrored requests. Restore
            # the critical request without sending input during focus loss.
            if ctx.pause_event.is_set():
                ctx.request_critical_danger()
                return False
            escaped = False
            try:
                escape_kwargs = {"reason": "critical_hunt"}
                if prefer_safe_key:
                    escape_kwargs["prefer_safe_key"] = True
                escaped = bool(self._teleport.danger_teleport(**escape_kwargs))
            except Exception as exc:
                ctx.logger.behavior(
                    f"[DANGER] critical hunting teleport failed: {exc}"
                )
            if escaped:
                ctx.logger.behavior(
                    "[DANGER] critical hunting escape succeeded"
                )
                return True
            # Keep the event pending for a retry rather than resuming combat
            # after an unavailable or interrupted teleport.
            ctx.request_critical_danger()
            return False
        finally:
            end_critical = getattr(ctx, "end_critical_escape_ops", None)
            if callable(end_critical):
                end_critical()
            else:
                end_escape = getattr(ctx, "end_danger_escape", None)
                if callable(end_escape):
                    end_escape()
                # Legacy contexts without end_critical_escape_ops release the
                # borrowed sit gate here. A random fly-wing escape landing is
                # not verified safe, so startup actions must wait for the
                # first discovery scan rather than trusting the area clear.
                ctx.end_sit_ops(trusted_clear=False)

    def run(self) -> None:
        ctx = self._ctx
        while not ctx.is_stopped():
            if not self.process_pending():
                # Poll at the danger detector's cadence so a critical drop is
                # answered as fast as the detector can observe it.
                ctx.stop_event.wait(CRITICAL_DANGER_POLL_INTERVAL_S)
