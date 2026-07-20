"""Hunt mode strategies — Teleport, Hybrid (placeholder), and Walk.

Each strategy encapsulates the no-target behaviour for a hunt mode,
extracted from HuntModeController to satisfy the Open/Closed Principle.
New hunt modes can be added by implementing a new strategy without
modifying the controller.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod

from pybot.runtime.constants import LOG_REPEAT_INTERVAL_MS
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.workers.worker_contexts import HuntModeControllerContext


class HuntModeStrategy(ABC):
    """Shared helpers and state: discovery tracking, logging, area-clear checks.

    Concrete strategies implement ``_handle_no_targets_impl()`` with
    mode-specific behaviour.  The base class handles the common guard
    logic (pause/stop checks, attackable tracks).
    """

    def __init__(
        self,
        ctx: HuntModeControllerContext,
        input_backend: InputBackend,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._discovery_area_epoch: int | None = None
        self._discovery_confirmed_clear = False
        self._last_no_target_blocked_log_ms = 0
        self._last_no_target_wait_reason: str | None = None
        self._last_discovery_fail_reason = ""
        self._last_discovery_fail_log_ms = 0
        self._lock = threading.Lock()

    # ── Public interface (called by HuntModeController) ──────────

    @property
    def discovery_since_reset(self) -> bool:
        """True after discovery has completed for the current area epoch."""
        with self._lock:
            return self._discovery_area_epoch == self._ctx.tracks.area_epoch

    @property
    def discovery_confirmed_clear(self) -> bool:
        """True when the latest discovery scan saw zero living candidates.

        Uses scan detections (pre-dedup), not post-reconcile alive tracks, so
        corpse heat matched only to death ghosts still blocks teleport.
        """
        with self._lock:
            return (
                self._discovery_confirmed_clear
                and self._discovery_area_epoch == self._ctx.tracks.area_epoch
            )

    def on_area_reset(self) -> None:
        """Reset per-area state (discovery flag, log throttles).

        Subclasses may extend to reset mode-specific timers.
        """
        with self._lock:
            self._discovery_area_epoch = None
            self._discovery_confirmed_clear = False
            self._last_no_target_blocked_log_ms = 0
            self._last_no_target_wait_reason = None
        self._on_area_reset_unlocked()

    def _on_area_reset_unlocked(self) -> None:
        """Hook for subclasses after shared state is cleared under lock."""

    def note_discovery_scan_completed(
        self,
        *,
        living_count: int,
        added_count: int,
        area_epoch: int,
    ) -> None:
        """Record a successful discovery scan for *area_epoch*."""
        del added_count
        with self._lock:
            if area_epoch != self._ctx.tracks.area_epoch:
                return
            self._discovery_area_epoch = area_epoch
            self._discovery_confirmed_clear = living_count == 0

    def note_discovery_scan_failed(self, reason: str) -> None:
        """Record a failed discovery scan."""
        if not reason:
            return
        now = monotonic_ms()
        with self._lock:
            if (
                reason == self._last_discovery_fail_reason
                and now - self._last_discovery_fail_log_ms < LOG_REPEAT_INTERVAL_MS
            ):
                return
            self._last_discovery_fail_reason = reason
            self._last_discovery_fail_log_ms = now
        self._ctx.logger.behavior(f"[DISCOVERY] scan failed reason={reason}")

    def on_no_attackable_targets(self) -> bool:
        """Handle the case when no attackable targets exist.

        Performs common guard checks (pause/stop, alive tracks)
        then dispatches to the mode-specific implementation.

        Returns:
            True if the bot took a mode-specific action (teleport, etc.).
        """
        ctx = self._ctx
        if ctx.pause_event.is_set() or ctx.stop_event.is_set():
            self._log_no_target("skip", "bot_not_running")
            return False

        now = monotonic_ms()
        if ctx.tracks.has_alive_tracks(now):
            self._log_no_target("wait", "alive_tracks")
            return False

        return self._handle_no_targets_impl()

    @abstractmethod
    def _handle_no_targets_impl(self) -> bool:
        """Mode-specific no-target behaviour (teleport, walk-wait, …).

        Called by ``on_no_attackable_targets()`` after common guards pass.
        """
        ...

    # ── Shared helpers used by concrete strategies ───────────────

    def _build_no_target_context(self) -> dict[str, object]:
        ctx = self._ctx
        now = monotonic_ms()
        area = ctx.tracks.get_area_clear_candidate(now)
        return {
            "alive_count": area.alive_count,
            "area_clear": area.clear,
            "has_discovery_since_reset": self.discovery_since_reset,
            "discovery_confirmed_clear": self.discovery_confirmed_clear,
        }

    def _log_no_target(
        self,
        decision: str,
        reason: str,
        context: dict | None = None,
    ) -> None:
        # Repeated idle "wait" lines are logged once per reason until area reset.
        # Teleport/skip decisions always log since they're infrequent.
        if decision == "wait":
            with self._lock:
                if reason == self._last_no_target_wait_reason:
                    return
                self._last_no_target_wait_reason = reason
            self._ctx.logger.behavior(f"[MODE] waiting reason={reason}")

        ctx = self._ctx
        ctx_data = context or self._build_no_target_context()
        ctx.validation.log_no_target_decision(
            decision,
            reason,
            alive_count=int(ctx_data["alive_count"]),
            area_clear=bool(ctx_data["area_clear"]),
            has_discovery_since_reset=bool(
                ctx_data["has_discovery_since_reset"]
            ),
        )

    def _log_no_target_blocked(self, reason: str) -> None:
        now = monotonic_ms()
        if now - self._last_no_target_blocked_log_ms < LOG_REPEAT_INTERVAL_MS:
            return
        self._last_no_target_blocked_log_ms = now
        self._ctx.logger.behavior(f"[MODE] no-target blocked reason={reason}")


class TeleportStrategy(HuntModeStrategy):
    """Teleport when area is clear of mobs."""

    def _handle_no_targets_impl(self) -> bool:
        ctx = self._ctx
        context = self._build_no_target_context()

        # Require a discovery scan that left zero alive tracks. Tracks can drop
        # earlier (lost) while mobs are still on screen; without this gate we
        # would teleport on a stale "discovery ran once" flag.
        if not self.discovery_confirmed_clear:
            reason = (
                "no_discovery_yet"
                if not self.discovery_since_reset
                else "discovery_not_clear"
            )
            # Don't sit on the 1s discovery cadence — ask for a scan now.
            if not ctx.discovery_suspend.is_set():
                ctx.discovery_wake.set()
            self._log_no_target_blocked(reason)
            self._log_no_target("wait", reason, context)
            return False

        area = ctx.tracks.get_area_clear_candidate()
        if not area.clear:
            self._log_no_target_blocked(area.reason)
            self._log_no_target("wait", area.reason, context)
            return False

        if not ctx.config.teleport_scan_code:
            self._log_no_target("wait", "no_teleport_key", context)
            return False

        # Suspend discovery before claim so the 1s cadence cannot scan during
        # teleport settle and confirm clear on a loading/empty frame.
        ctx.discovery_suspend.set()
        ctx.discovery_wake.clear()

        # Claim under the tracks lock before input so a concurrent discovery
        # reconcile cannot spawn tracks into the area we are leaving.
        if not ctx.tracks.try_claim_clear_for_teleport():
            ctx.discovery_suspend.clear()
            self._log_no_target_blocked("alive_tracks")
            self._log_no_target("wait", "alive_tracks", context)
            return False
        ctx.policy.reset()
        ctx.validation.log_area_reset("pre_teleport")
        self.on_area_reset()

        ctx.logger.behavior(
            f"[MODE] teleport area_clear tracks={area.alive_count}"
        )
        self._log_no_target("teleport", "area_clear", context)

        try:
            self._input.teleport_key(ctx.config.teleport_scan_code)
        except Exception as exc:
            ctx.logger.behavior(
                f"[MODE] teleport input error: {exc}"
            )
            self._release_discovery_after_teleport()
            return False
        ctx.overlay.increment_teleports()
        if not ctx.wait_unless_stopped(ctx.config.teleport_duration_ms / 1000.0):
            self._release_discovery_after_teleport()
            return False
        self._release_discovery_after_teleport()
        return True

    def _release_discovery_after_teleport(self) -> None:
        """Allow discovery again and wake it for the post-settle scan."""
        ctx = self._ctx
        ctx.discovery_suspend.clear()
        ctx.discovery_wake.set()


class HybridStrategy(HuntModeStrategy):
    """Placeholder hunt mode — no hybrid behaviour yet."""

    def _handle_no_targets_impl(self) -> bool:
        self._log_no_target("wait", "hybrid_not_implemented")
        return False


class WalkStrategy(HuntModeStrategy):
    """Wait for mobs to path into detection range (no teleport)."""

    def __init__(
        self,
        ctx: HuntModeControllerContext,
        input_backend: InputBackend,
    ) -> None:
        super().__init__(ctx, input_backend)
        self._walk_idle_start_ms = 0

    def _on_area_reset_unlocked(self) -> None:
        self._walk_idle_start_ms = 0

    def _handle_no_targets_impl(self) -> bool:
        # The base guard already returned when any alive track exists, so here
        # the area is empty. Walk mode never teleports — it only waits and logs.
        ctx = self._ctx
        now = monotonic_ms()

        # Start the idle timer once, on first entry into the no-target state.
        with self._lock:
            if not self._walk_idle_start_ms:
                self._walk_idle_start_ms = now
                ctx.logger.behavior("[MODE] walk mode — waiting for mobs to appear")
            walk_idle_start_ms = self._walk_idle_start_ms

        if not self.discovery_since_reset:
            idle_seconds = (now - walk_idle_start_ms) // 1000
            if idle_seconds > 0 and idle_seconds % 15 == 0:
                ctx.logger.behavior(
                    f"[MODE] walk waiting for first discovery elapsed={idle_seconds}s"
                )
            self._log_no_target("wait", "walk_no_discovery_yet")
            return False

        # Discovery has run and the area is empty — wait for mobs to path in.
        self._log_no_target("wait", "walk_area_clear")
        return False
