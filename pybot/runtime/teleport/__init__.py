"""TeleportController — every teleport concern in one place.

Responsibilities
----------------
* **Key selection** — mode/area: creamy TP first, wing key next.
  Danger/critical: wing key when Teleport Key is assigned, else creamy.
* **Execution** — press teleport key, wait for settle, track wings/overlay.
* **Danger teleport** — the sit worker's urgent escape primitive.
* **Mode teleport** — discovery-gated teleport with suspend/release (for TeleportStrategy).
  Hunt-mode teleports wait when Sit On Low Sp is enabled and SP is unread or
  below the sit threshold, so recovery can start instead of chaining keys.

Usage
-----
Create once per hunt session and inject into workers and the
TeleportStrategy. DangerDetector only observes HP/visual state; the sit worker
owns danger escape.
"""

from __future__ import annotations

import time

from pybot.runtime.constants import (
    HP_POST_TELEPORT_HEAL_S,
    SIT_LOW_SP_RATIO,
    SIT_SP_POLL_INTERVAL_S,
)
from pybot.runtime.event_utils import event_is_set
from pybot.runtime.input.input_backend import InputBackend
from pybot.game_state import PlayerVitals


class TeleportController:
    """Centralises every teleport concern — key choice, press, settle, cleanup."""

    def __init__(
        self,
        ctx,
        input_backend: InputBackend,
        hunt_mode=None,
        *,
        vitals: PlayerVitals | None = None,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._hunt_mode = hunt_mode
        self._vitals = vitals

    def bind_hunt_mode(self, hunt_mode) -> None:
        """Wire the hunt-mode area-reset notification after both exist.

        ``TeleportStrategy`` needs this controller before the mode controller
        exists, so the composition root binds the mode once ``create_hunt_mode``
        has returned. Only ``on_area_reset`` is consumed.
        """
        self._hunt_mode = hunt_mode

    def _notify_area_reset(self) -> None:
        """Notify the bound hunt mode that the area changed."""
        if self._hunt_mode is not None:
            self._hunt_mode.on_area_reset()

    # ── Key selection ────────────────────────────────────────────

    def active_scan_code(self) -> int:
        """Creamy TP first, wing key next (mode / area-clear teleports)."""
        cfg = self._ctx.config
        if self._creamy_assigned(cfg):
            return int(cfg.creamy_tp_scan_code)
        if self._wing_key_assigned(cfg):
            return int(cfg.teleport_scan_code)
        return 0

    def active_button(self) -> str:
        """Creamy TP first, wing key next (mode / area-clear teleports)."""
        cfg = self._ctx.config
        if self._creamy_assigned(cfg):
            return cfg.creamy_tp_button
        return cfg.teleport_button

    def _scan_code_is_configured(self, scan_code: int) -> bool:
        """Return whether a teleport scan code still has an assigned key.

        Callers may pass an explicit code for a selected teleport path, but
        that must not bypass the same assignment checks used by key selection.
        This protects against stale runtime objects after a binding is cleared.
        """
        cfg = self._ctx.config
        return (
            (
                int(cfg.teleport_scan_code) == scan_code
                and bool((cfg.teleport_button or "").strip())
            )
            or (
                int(cfg.creamy_tp_scan_code) == scan_code
                and bool((cfg.creamy_tp_button or "").strip())
            )
        )

    @staticmethod
    def _wing_key_assigned(cfg) -> bool:
        """True when Teleport Key (fly wing) is bound to a real scan code."""
        return bool((cfg.teleport_button or "").strip()) and int(cfg.teleport_scan_code) > 0

    @staticmethod
    def _creamy_assigned(cfg) -> bool:
        """True when Creamy TP is bound to a real scan code."""
        return (
            bool((cfg.creamy_tp_button or "").strip())
            and int(cfg.creamy_tp_scan_code) > 0
        )

    def danger_scan_code(self) -> int:
        """Wing (Teleport) key when assigned; otherwise Creamy TP.

        Urgent/critical escapes use fly wings only if Teleport Key is set.
        If wing is unset, creamy must still fire — never a no-op escape.
        Once GetFlyWings reports that storage has no wings left
        (``fly_wings_exhausted``), the wing key is a no-op in-game, so the
        critical escape switches to the Creamy TP key while one is bound.
        """
        cfg = self._ctx.config
        wings_exhausted = getattr(self._ctx, "fly_wings_exhausted", None) is True
        if wings_exhausted and self._creamy_assigned(cfg):
            return int(cfg.creamy_tp_scan_code)
        if self._wing_key_assigned(cfg):
            return int(cfg.teleport_scan_code)
        if self._creamy_assigned(cfg):
            return int(cfg.creamy_tp_scan_code)
        return 0

    def hunt_teleport_blocked_reason(self) -> str | None:
        """Return why a hunt-mode teleport must wait, or ``None`` to proceed.

        Sit-on-low-SP recovery only starts from a readable sample below
        ``SIT_LOW_SP_RATIO``. Hunt teleports clear that sample for the landing
        epoch, so chaining area-clear teleports before a fresh reading would
        skip recovery and keep pressing the teleport key. Danger and sit
        placement teleports do not use this gate.
        """
        cfg = self._ctx.config
        if getattr(cfg, "sit_on_low_sp", False) is not True:
            return None
        if not str(getattr(cfg, "sit_on_low_sp_button", "") or "").strip():
            return None
        if int(getattr(cfg, "sit_on_low_sp_scan_code", 0) or 0) <= 0:
            return None
        if self._vitals is None:
            return "sp_unknown"
        sp, sp_max = self._vitals.sp_pair()
        if sp is None or sp_max is None or sp_max <= 0:
            return "sp_unknown"
        if (sp / sp_max) < SIT_LOW_SP_RATIO:
            return "low_sp"
        return None

    def danger_button(self) -> str:
        """Wing (Teleport) key when assigned; otherwise Creamy TP.

        Mirrors :meth:`danger_scan_code`: the critical escape uses Creamy TP
        once fly wings are exhausted and a creamy binding exists.
        """
        cfg = self._ctx.config
        wings_exhausted = getattr(self._ctx, "fly_wings_exhausted", None) is True
        if wings_exhausted and self._creamy_assigned(cfg):
            return cfg.creamy_tp_button
        if self._wing_key_assigned(cfg):
            return cfg.teleport_button
        return cfg.creamy_tp_button

    # ── Press + settle ───────────────────────────────────────────

    def teleport_once(self, scan_code: int | None = None) -> bool:
        """Press a teleport key, clear tracks immediately, then wait to settle.

        Args:
            scan_code: Key to press. Defaults to :meth:`active_scan_code`.

        Returns ``False`` if no key is configured or the wait was interrupted.
        """
        cfg = self._ctx.config
        tp = int(scan_code) if scan_code is not None else self.active_scan_code()
        if tp <= 0 or not self._scan_code_is_configured(tp):
            return False
        teleport_started = time.monotonic()
        try:
            accepted = self._input.teleport_key(tp)
        except Exception as exc:
            self._ctx.logger.behavior(f"[TP] input error: {exc}")
            return False
        if accepted is False:
            self._ctx.logger.behavior(
                f"[TP] key rejected scan={tp} — transition not started"
            )
            return False
        # The key was accepted: remove every old-area track immediately. Do
        # not spend any further transition work before this reset. Tracking and
        # discovery may still finish work captured before the key, but the
        # advanced area epoch rejects those late publications. This is
        # deliberately fail-closed: even an interrupted settle cannot leave
        # old targets actionable.
        self._reset_tracking("teleport", log_tag="TP")
        attack_wake = getattr(self._ctx, "attack_wake", None)
        transition_epoch = None
        if self._vitals is not None:
            begin_epoch = getattr(self._vitals, "begin_observation_epoch", None)
            if callable(begin_epoch):
                # Keep one quarantine token for the complete key→landing
                # transition. Readers captured before the key, during settle,
                # and before this completion all fail the same token check.
                transition_epoch = begin_epoch()
        if attack_wake is not None:
            attack_wake.clear()

        # Only decrement the fly-wing counter when the wing key was used.
        if tp == cfg.teleport_scan_code and cfg.teleport_scan_code > 0:
            self._ctx.note_teleport_for_wings()
        self._ctx.overlay.increment_teleports()
        settled = self._wait_for_settle(cfg.teleport_duration_ms / 1000.0)
        # PlayerVitals is app-scoped and reused across hunt sessions. Leave
        # the epoch quarantined only while landing frames can still publish;
        # Stop during settle must reopen it or later HP/SP reads stay dead.
        # Reset the danger baseline while HP is still quarantined so the
        # first post-landing sample is a new baseline, not a phantom drop
        # against the previous area's HP.
        danger = self._ctx.danger_detector
        if danger is not None:
            danger.reset_after_teleport(teleport_started)
        if self._vitals is not None:
            complete_epoch = getattr(
                self._vitals, "complete_observation_epoch", None
            )
            if callable(complete_epoch):
                complete_epoch(transition_epoch)
        if settled:
            self._ctx.mark_post_teleport_heal(HP_POST_TELEPORT_HEAL_S)
        return settled

    def retry_post_teleport_heal(self) -> bool:
        """Teleport again so post-teleport healing can be retried safely."""
        ctx = self._ctx
        if self._escape_in_flight():
            return False
        tp = self.active_scan_code()
        if tp <= 0:
            ctx.logger.behavior(
                "[HEAL] retry teleport skipped — no teleport key configured"
            )
            return False
        return self._run_area_transition(
            lambda: self.teleport_once(scan_code=tp),
        )

    def _wait_for_settle(self, timeout_s: float) -> bool:
        """Wait for a teleport already in flight to finish landing.

        A critical request can arrive after the teleport key was accepted. It
        must not turn that in-flight teleport into a reported failure: retrying
        the key before the client has landed can send duplicate teleports and
        skip the post-teleport damage baseline reset. The detector preserves
        damage observed during this wait, and the owning gameplay sequence
        handles that fresh request immediately after the landing boundary.

        Once input was accepted, a user pause also must not make this teleport
        look failed. If the normal wait is interrupted by pause, finish the
        remaining settle window while honoring stop; a later gameplay tick will
        observe the pause and defer its next action.
        """
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            if event_is_set(self._ctx.stop_event) is True:
                return False
            paused = event_is_set(self._ctx.pause_event) is True
            if paused:
                self._ctx.stop_event.wait(
                    min(SIT_SP_POLL_INTERVAL_S, remaining)
                )
                continue
            waited = self._ctx.wait_unless_stopped(remaining)
            if waited:
                return True
            # A false result can be caused by a pause racing with the check;
            # finish the already-issued teleport's settle after that pause.
            if event_is_set(self._ctx.pause_event) is True:
                continue
            # A false result without pause means the wait was stopped. Preserve
            # that failure result rather than retrying an already-issued key.
            return False


    def _escape_in_flight(self) -> bool:
        """True while any urgent danger escape owns the teleport key.

        Only the escape itself may press a teleport key while it is in
        flight; every other teleport path (sit placement, storage clear,
        mode teleport) must yield so the escape key is never queued behind
        or alongside a competing teleport.
        """
        return event_is_set(self._ctx.danger_escape_active) is True

    def teleport_once_for_sit(self, *, log_tag: str = "SIT") -> bool:
        """Teleport exactly once before sitting to recover SP.

        Low-SP recovery must not keep searching for a perfectly empty screen:
        one fresh area is enough, then the sit worker owns the spot. Reset all
        stale area state after the single teleport so the next hunt generation
        cannot act on tracks from the previous screen.

        If a critical escape claims while this placement is already settling,
        both teleports fire (placement + escape key). That window is narrow
        and self-correcting — the escape wins and the recovery re-sits — but
        two consumables may be spent in that one race.
        """
        if self._escape_in_flight():
            self._ctx.logger.behavior(
                "[SIT] placement teleport skipped — critical escape in flight"
            )
            return False
        if not self.teleport_once():
            return False
        return True

    # ── Danger teleport ──────────────────────────────────────────

    def danger_teleport(
        self,
        reason: str = "",
        *,
        prefer_safe_key: bool = False,
    ) -> bool:
        """Press the danger teleport key and clear tracks after.

        Suspends discovery for the claim → key → settle window so a concurrent
        scan cannot confirm clear on a loading frame. Track reset is owned by
        :meth:`teleport_once` immediately after the key is accepted, before
        landing completes.

        ``prefer_safe_key`` selects the same key used for sit/storage placement
        (creamy / save point first) instead of the urgent random fly wing. A
        seated recovery escape must land somewhere the character can sit; the
        random wing can drop it back next to mobs and cause repeated
        escape→sit cycles. The hunting critical escape keeps the random wing
        to break combat immediately.
        """
        ctx = self._ctx
        prefix = f"{reason} " if reason else ""
        if prefer_safe_key:
            tp = self.active_scan_code()
            key_name = self.active_button() or "(unset)"
        else:
            tp = self.danger_scan_code()
            key_name = self.danger_button() or "(unset)"
        if tp <= 0:
            ctx.logger.behavior(
                f"[DANGER] {prefix}no teleport key configured "
                f"({key_name!r}) — cannot escape"
            )
            return False
        ctx.logger.behavior(
            f"[DANGER] {prefix}key={key_name!r} scan={tp} — teleporting"
        )
        # Hold the lifecycle boundary through input, settle, and reset.
        # No-target decisions cannot observe the old strategy state while
        # the gameplay owner is already in the danger transition.
        # ``teleport_once`` clears tracks under the same re-entrant lock.
        return self._run_area_transition(
            lambda: self.teleport_once(scan_code=tp),
        )

    # ── Mode teleport (for TeleportStrategy) ──────────────────────

    def mode_teleport(self) -> bool:
        """Discovery-gated teleport: suspend, claim clear, press, release.

        The caller must already have checked ``discovery_confirmed_clear``
        and ``area_clear`` before calling.

        Returns ``True`` on successful teleport.
        """
        ctx = self._ctx

        # A critical escape owns the teleport key; the mode transition must
        # yield even after its escape request was consumed mid-settle.
        if self._escape_in_flight():
            ctx.logger.behavior(
                "[MODE] teleport skipped — critical escape in flight"
            )
            return False

        blocked = self.hunt_teleport_blocked_reason()
        if blocked is not None:
            ctx.logger.behavior(f"[MODE] teleport skipped — {blocked}")
            return False

        # Suspend discovery so the 1s cadence cannot scan during teleport
        # settle and confirm clear on a loading / empty frame. The clear claim
        # remains inside the same transition lock before the key is pressed.
        def transition() -> bool:
            # Validate under the tracks lock without mutating the area. The
            # actual reset belongs to teleport_once and occurs immediately
            # after the key is accepted; a rejected key must not advance the
            # area epoch or discard the current screen's state.
            if not ctx.tracks.can_claim_clear_for_teleport():
                return False
            # ``teleport_once`` owns wake cleanup together with the accepted
            # input reset. Do not clear it during this read-only admission
            # check: a rejected key must leave the current area untouched.
            tp_button = self.active_button()
            ctx.logger.behavior(
                f"[MODE] teleport key={tp_button!r} "
                f"wingsExhausted={ctx.fly_wings_exhausted}"
            )
            return self.teleport_once()

        return self._run_area_transition(transition)

    # ── Internal helpers ─────────────────────────────────────────

    def _run_area_transition(self, action) -> bool:
        """Run one teleport transition with one shared suspend/lock boundary.

        All teleport entry points must use the same boundary: discovery stays
        out of loading frames, and track/policy publication cannot interleave
        with the input, settle, and reset sequence. Keeping this ownership in
        one helper removes three subtly divergent wrappers without changing
        detector work or transition ordering.
        """
        ctx = self._ctx
        ctx.discovery_suspend.set()
        ctx.discovery_wake.clear()
        try:
            with ctx.area_transition_lock:
                return bool(action())
        finally:
            ctx.discovery_suspend.clear()
            ctx.discovery_wake.set()

    def _reset_tracking(self, reason: str, *, log_tag: str) -> None:
        """Clear tracks/policy/overlay and hunt-mode flags after teleport."""
        ctx = self._ctx
        with ctx.area_transition_lock:
            self._notify_area_reset()
            ctx.area_reset(reason)
        ctx.overlay.set_track_stats(track_count=0, alive_count=0)
        ctx.overlay.set_track_positions([])
        ctx.logger.behavior(f"[{log_tag}] tracking reset reason={reason}")
