"""TeleportController — every teleport concern in one place.

Responsibilities
----------------
* **Key selection** — mode/area: creamy TP first, wing key next.
  Danger/critical: wing key when Teleport Key is assigned, else creamy.
* **Execution** — press teleport key, wait for settle, track wings/overlay.
* **Danger teleport** — the sit worker's urgent escape primitive.
* **Area clear** — scan-loop that teleports until discovery sees zero mobs.
* **Quiet area** — clear + idle + re-scan for recovery/sit placement.
* **Mode teleport** — discovery-gated teleport with suspend/release (for TeleportStrategy).

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
    SIT_IDLE_BEFORE_SIT_S,
    SIT_SP_POLL_INTERVAL_S,
)
from pybot.runtime.detection.discovery_filter import filter_scan_candidates
from pybot.runtime.event_utils import event_is_set
from pybot.runtime.input.input_backend import InputBackend


class TeleportController:
    """Centralises every teleport concern — key choice, press, settle, cleanup."""

    def __init__(
        self,
        ctx,
        input_backend: InputBackend,
        hunt_mode=None,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._hunt_mode = hunt_mode

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
        return cfg.creamy_tp_scan_code or cfg.teleport_scan_code

    def active_button(self) -> str:
        """Creamy TP first, wing key next (mode / area-clear teleports)."""
        cfg = self._ctx.config
        if cfg.creamy_tp_scan_code > 0 and cfg.creamy_tp_button:
            return cfg.creamy_tp_button
        return cfg.teleport_button

    @staticmethod
    def _wing_key_assigned(cfg) -> bool:
        """True when Teleport Key (fly wing) is bound to a real scan code."""
        return bool((cfg.teleport_button or "").strip()) and int(cfg.teleport_scan_code) > 0

    def danger_scan_code(self) -> int:
        """Wing (Teleport) key when assigned; otherwise Creamy TP.

        Urgent/critical escapes use fly wings only if Teleport Key is set.
        If wing is unset, creamy must still fire — never a no-op escape.
        """
        cfg = self._ctx.config
        if self._wing_key_assigned(cfg):
            return int(cfg.teleport_scan_code)
        return int(cfg.creamy_tp_scan_code)

    def danger_button(self) -> str:
        """Wing (Teleport) key when assigned; otherwise Creamy TP."""
        cfg = self._ctx.config
        if self._wing_key_assigned(cfg):
            return cfg.teleport_button
        return cfg.creamy_tp_button

    # ── Press + settle ───────────────────────────────────────────

    def teleport_once(self, scan_code: int | None = None) -> bool:
        """Press a teleport key, wait for settle, track wings/overlay.

        Args:
            scan_code: Key to press. Defaults to :meth:`active_scan_code`.

        Returns ``False`` if no key is configured or the wait was interrupted.
        """
        cfg = self._ctx.config
        tp = int(scan_code) if scan_code is not None else self.active_scan_code()
        if tp <= 0:
            return False
        teleport_started = time.monotonic()
        try:
            self._input.teleport_key(tp)
        except Exception as exc:
            self._ctx.logger.behavior(f"[TP] input error: {exc}")
            return False
        # Only decrement the fly-wing counter when the wing key was used.
        if tp == cfg.teleport_scan_code and cfg.teleport_scan_code > 0:
            self._ctx.note_teleport_for_wings()
        self._ctx.overlay.increment_teleports()
        settled = self._wait_for_settle(cfg.teleport_duration_ms / 1000.0)
        if settled:
            # Every successful teleport starts a new area. Clear only damage
            # observed before this teleport. Damage recorded during settle
            # remains a fresh danger signal.
            self._ctx.mark_post_teleport_heal(HP_POST_TELEPORT_HEAL_S)
            danger = self._ctx.danger_detector
            if danger is not None:
                danger.reset_after_teleport(teleport_started)
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
        ctx.discovery_suspend.set()
        ctx.discovery_wake.clear()
        try:
            with ctx.area_transition_lock:
                if not self.teleport_once(scan_code=tp):
                    return False
                self._reset_tracking(
                    "post_teleport_heal_retry",
                    log_tag="HEAL",
                )
                return True
        finally:
            ctx.discovery_suspend.clear()
            ctx.discovery_wake.set()

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
        self._reset_tracking(f"{log_tag.lower()}_teleport", log_tag=log_tag)
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
        scan cannot confirm clear on a loading frame.

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
        ctx.discovery_suspend.set()
        ctx.discovery_wake.clear()
        try:
            # Hold the lifecycle boundary through input, settle, and reset.
            # No-target decisions cannot observe the old strategy state while
            # the critical worker is already in the danger transition.
            with ctx.area_transition_lock:
                if not self.teleport_once(scan_code=tp):
                    return False
                # Publish the strategy marker and track epoch as one lifecycle
                # transaction. The lock is re-entrant because ctx.area_reset
                # also protects direct callers with the same boundary.
                self._notify_area_reset()
                ctx.area_reset(reason="danger_teleport")
                return True
        finally:
            ctx.discovery_suspend.clear()
            ctx.discovery_wake.set()

    # ── Area clear loops ─────────────────────────────────────────

    def teleport_until_clear(self, log_tag: str) -> bool:
        """Teleport + settle until discovery finds zero living mobs.

        Returns ``True`` when the area is clear, ``False`` if stopped.
        """
        tp = self.active_scan_code()
        if tp <= 0:
            key_name = self.active_button() or "(unset)"
            self._ctx.logger.behavior(
                f"[{log_tag}] no teleport key configured ({key_name!r}) — "
                "cannot clear area"
            )
            return False

        while not self._ctx.is_stopped():
            if self._escape_in_flight():
                return False
            if (
                self._danger_request_is_set() is True
                or self._critical_request_is_set() is True
            ):
                return False
            living = self._scan_living_count()
            if living is None:
                self._ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                continue
            if living == 0:
                self._ctx.logger.behavior(
                    f"[{log_tag}] discovery sees no mobs"
                )
                self._reset_tracking(f"{log_tag.lower()}_clear", log_tag=log_tag)
                return True

            self._ctx.logger.behavior(
                f"[{log_tag}] discovery living={living} — teleport before UI"
            )
            # Danger has priority even if it arrives after the scan but before
            # the storage clear teleport. Never issue a competing teleport
            # once the urgent request is visible.
            if (
                self._escape_in_flight()
                or self._danger_request_is_set() is True
                or self._critical_request_is_set() is True
            ):
                return False
            if not self.teleport_once():
                return False
            self._reset_tracking(
                f"{log_tag.lower()}_teleport", log_tag=log_tag
            )
        return False

    def _critical_request_is_set(self) -> bool | None:
        """Return the pending critical-danger event state."""
        return event_is_set(self._ctx.critical_danger_requested)

    def _danger_request_is_set(self) -> bool | None:
        """Return the pending seated-danger event state."""
        return event_is_set(self._ctx.danger_sit_requested)

    def teleport_until_quiet(
        self,
        log_tag: str,
        idle_s: float = SIT_IDLE_BEFORE_SIT_S,
    ) -> bool:
        """Clear area, idle, then re-scan, aborting promptly for danger."""
        while not self._ctx.is_stopped():
            if self._escape_in_flight():
                return False
            if (
                self._danger_request_is_set() is True
                or self._critical_request_is_set() is True
            ):
                return False
            if not self.teleport_until_clear(log_tag=log_tag):
                return False
            self._ctx.logger.behavior(
                f"[{log_tag}] area clear — idle {idle_s:.0f}s before proceed"
            )

            deadline = time.monotonic() + idle_s
            while True:
                if self._escape_in_flight():
                    return False
                if (
                    self._danger_request_is_set() is True
                    or self._critical_request_is_set() is True
                ):
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                # Check at the worker cadence so a damage request cannot wait
                # through the whole idle window.
                wait_s = min(SIT_SP_POLL_INTERVAL_S, remaining)
                if not self._ctx.wait_unless_stopped(wait_s):
                    return False

            if self._escape_in_flight():
                return False
            if (
                self._danger_request_is_set() is True
                or self._critical_request_is_set() is True
            ):
                return False
            living = self._scan_living_count()
            if living is None:
                self._ctx.logger.behavior(
                    f"[{log_tag}] post-idle scan failed — clear again"
                )
                continue
            if living == 0:
                self._ctx.logger.behavior(f"[{log_tag}] still clear after idle")
                return True
            self._ctx.logger.behavior(
                f"[{log_tag}] mobs during idle (living={living}) — clear again"
            )
        return False

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

        # Suspend discovery so the 1s cadence cannot scan during teleport
        # settle and confirm clear on a loading / empty frame.
        ctx.discovery_suspend.set()
        ctx.discovery_wake.clear()

        try:
            with ctx.area_transition_lock:
                # Claim under the tracks lock before input so a concurrent
                # discovery reconcile cannot spawn tracks into the area we are
                # leaving. The strategy reset stays in this same transaction.
                if not ctx.tracks.try_claim_clear_for_teleport():
                    return False

                ctx.policy.reset()
                ctx.validation.log_area_reset("pre_teleport")
                self._notify_area_reset()

                tp_button = self.active_button()
                ctx.logger.behavior(
                    f"[MODE] teleport key={tp_button!r} "
                    f"wingsExhausted={ctx.fly_wings_exhausted}"
                )
                ok = self.teleport_once()
                return ok
        finally:
            ctx.discovery_suspend.clear()
            ctx.discovery_wake.set()

    # ── Internal helpers ─────────────────────────────────────────

    def no_visible_mobs_now(self) -> bool:
        """Return true only when an immediate discovery scan sees zero mobs.

        This is intentionally a fresh detector read rather than the cached
        strategy flag. Storage uses it immediately before opening inventory so
        a mob appearing after the quiet-area wait cannot be treated as safe.
        """
        living = self._scan_living_count()
        return living == 0

    def _scan_living_count(self) -> int | None:
        """Run one discovery scan; return filtered living count or ``None``."""
        ctx = self._ctx
        if not ctx.capture.is_valid():
            return None
        roi = ctx.capture.get_hunt_roi()
        if roi is None:
            return None
        frame = ctx.capture.capture_roi(roi)
        if frame is None or frame.size == 0:
            return None
        scan = ctx.detector.discover_frame(frame, roi)
        if not scan.ok:
            return None
        filtered = filter_scan_candidates(scan.detections)
        return len(filtered)

    def _reset_tracking(self, reason: str, *, log_tag: str) -> None:
        """Clear tracks/policy/overlay and hunt-mode flags after teleport."""
        ctx = self._ctx
        with ctx.area_transition_lock:
            self._notify_area_reset()
            ctx.area_reset(reason)
        ctx.overlay.set_track_stats(track_count=0, alive_count=0)
        ctx.overlay.set_track_positions([])
        ctx.logger.behavior(f"[{log_tag}] tracking reset reason={reason}")
