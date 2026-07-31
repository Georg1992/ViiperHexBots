"""TeleportController — every teleport concern in one place.

Responsibilities
----------------
* **Key selection** — mode/area: creamy TP first, wing key next.
  Danger/critical: wing key first when assigned, creamy next.
* **Execution** — press teleport key, wait for settle, track wings/overlay.
* **Danger teleport** — for DangerDetector (critical HP only).
* **Area clear** — scan-loop that teleports until discovery sees zero mobs.
* **Quiet area** — clear + idle + re-scan (for sit/storage workers).
* **Mode teleport** — discovery-gated teleport with suspend/release (for TeleportStrategy).

Usage
-----
Create once per hunt session and inject into DangerDetector, workers,
and the TeleportStrategy.
"""

from __future__ import annotations

from pybot.runtime.constants import (
    HP_POST_TELEPORT_HEAL_S,
    SIT_IDLE_BEFORE_SIT_S,
    SIT_SP_POLL_INTERVAL_S,
)
from pybot.runtime.detection.discovery_filter import filter_scan_candidates
from pybot.runtime.input.input_backend import InputBackend


class TeleportController:
    """Centralises every teleport concern — key choice, press, settle, cleanup."""

    def __init__(self, ctx, input_backend: InputBackend, hunt_mode) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._hunt_mode = hunt_mode
        self._character_state = None

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

    def danger_scan_code(self) -> int:
        """Wing (Teleport) key first when assigned; creamy next.

        Critical / danger teleports must consume fly wings when the wing key
        is configured, not the creamy card.
        """
        cfg = self._ctx.config
        return cfg.teleport_scan_code or cfg.creamy_tp_scan_code

    def danger_button(self) -> str:
        """Wing (Teleport) key first when assigned; creamy next."""
        cfg = self._ctx.config
        if cfg.teleport_scan_code > 0 and cfg.teleport_button:
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
        try:
            self._input.teleport_key(tp)
        except Exception as exc:
            self._ctx.logger.behavior(f"[TP] input error: {exc}")
            return False
        # Only decrement the fly-wing counter when the wing key was used.
        if tp == cfg.teleport_scan_code and cfg.teleport_scan_code > 0:
            self._ctx.note_teleport_for_wings()
        self._ctx.overlay.increment_teleports()
        settled = self._ctx.wait_unless_stopped(
            cfg.teleport_duration_ms / 1000.0
        )
        if settled:
            # Aggro-free window after landing — heal may run immediately.
            self._ctx.mark_post_teleport_heal(HP_POST_TELEPORT_HEAL_S)
        return settled

    # ── Danger teleport ──────────────────────────────────────────

    def danger_teleport(self, reason: str = "") -> None:
        """Press the danger teleport key (wing first) and clear tracks after.

        Suspends discovery for the claim → key → settle window so a concurrent
        scan cannot confirm clear on a loading frame.
        """
        ctx = self._ctx
        prefix = f"{reason} " if reason else ""
        tp = self.danger_scan_code()
        key_name = self.danger_button() or "(unset)"
        if tp <= 0:
            ctx.logger.behavior(
                f"[DANGER] {prefix}no teleport key configured "
                f"({key_name!r}) — cannot escape"
            )
            return
        ctx.logger.behavior(
            f"[DANGER] {prefix}key={key_name!r} scan={tp} — teleporting"
        )
        ctx.discovery_suspend.set()
        ctx.discovery_wake.clear()
        try:
            self.teleport_once(scan_code=tp)
            ctx.area_reset(reason="danger_teleport")
            self._clear_character_threat()
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
            self.teleport_once()
            self._reset_tracking(
                f"{log_tag.lower()}_teleport", log_tag=log_tag
            )
        return False

    def teleport_until_quiet(
        self,
        log_tag: str,
        idle_s: float = SIT_IDLE_BEFORE_SIT_S,
    ) -> bool:
        """Clear area, idle, then re-scan.

        A single clear snapshot is not enough: mobs can walk into ROI (or first
        become detectable) during the post-clear idle before sit/storage UI.

        Returns ``True`` when still clear after the idle window.
        """
        while not self._ctx.is_stopped():
            if not self.teleport_until_clear(log_tag=log_tag):
                return False
            self._ctx.logger.behavior(
                f"[{log_tag}] area clear — idle {idle_s:.0f}s before proceed"
            )
            if not self._ctx.wait_unless_stopped(idle_s):
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

        # Suspend discovery so the 1s cadence cannot scan during teleport
        # settle and confirm clear on a loading / empty frame.
        ctx.discovery_suspend.set()
        ctx.discovery_wake.clear()

        try:
            # Claim under the tracks lock before input so a concurrent discovery
            # reconcile cannot spawn tracks into the area we are leaving.
            if not ctx.tracks.try_claim_clear_for_teleport():
                return False

            ctx.policy.reset()
            ctx.validation.log_area_reset("pre_teleport")
            self._hunt_mode.on_area_reset()
            self._clear_character_threat()

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
        ctx.area_reset(reason)
        self._hunt_mode.on_area_reset()
        self._clear_character_threat()
        ctx.overlay.set_track_stats(track_count=0, alive_count=0)
        ctx.overlay.set_track_positions([])
        ctx.logger.behavior(f"[{log_tag}] tracking reset reason={reason}")

    def _clear_character_threat(self) -> None:
        state = self._character_state
        if state is not None:
            state.clear_area_threat()
