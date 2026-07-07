"""Hunt mode strategies — Teleport and Walk (Strategy pattern).

Each strategy encapsulates the no-target behaviour for a hunt mode,
extracted from HuntModeController to satisfy the Open/Closed Principle.
New hunt modes can be added by implementing a new strategy without
modifying the controller.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime import overlay as hunt_overlay
from pybot.runtime.workers.worker_contexts import HuntModeControllerContext


class HuntModeStrategy(ABC):
    """Shared helpers and state: discovery tracking, logging, area-clear checks.

    Concrete strategies implement ``_handle_no_targets_impl()`` with
    mode-specific behaviour.  The base class handles the common guard
    logic (pause/stop checks, attackable tracks, alive-or-pending).
    """

    def __init__(
        self,
        ctx: HuntModeControllerContext,
        input_backend: InputBackend,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._discovery_since_reset = False
        self._last_no_target_blocked_log_ms = 0
        self._last_wait_known_log_ms = 0

    # ── Public interface (called by HuntModeController) ──────────

    @property
    def discovery_since_reset(self) -> bool:
        """True after at least one discovery scan has completed in this area."""
        return self._discovery_since_reset

    def on_area_reset(self) -> None:
        """Reset per-area state (discovery flag, log throttles).

        Subclasses may extend to reset mode-specific timers.
        """
        self._discovery_since_reset = False
        self._last_no_target_blocked_log_ms = 0
        self._last_wait_known_log_ms = 0

    def note_discovery_scan_completed(
        self, *, living_count: int, added_count: int
    ) -> None:
        """Record a successful discovery scan."""
        self._discovery_since_reset = True

    def note_discovery_scan_failed(self, reason: str) -> None:
        """Record a failed discovery scan."""
        if reason:
            self._ctx.logger.behavior(f"[DISCOVERY] scan failed reason={reason}")

    def on_no_attackable_targets(self) -> bool:
        """Handle the case when no attackable targets exist.

        Performs common guard checks (pause/stop, attackable tracks,
        alive-or-pending) then dispatches to the mode-specific
        implementation.

        Returns:
            True if the bot took a mode-specific action (teleport, etc.).
        """
        ctx = self._ctx
        if ctx.pause_event.is_set() or ctx.stop_event.is_set():
            self._log_no_target("skip", "bot_not_running")
            return False

        now = monotonic_ms()
        if ctx.tracks.has_attackable_tracks(now):
            self._log_no_target("wait", "attackable_tracks")
            return False

        alive_or_pending = ctx.tracks.get_alive_or_pending_count(now)
        if alive_or_pending > 0:
            self._log_wait_known(alive_or_pending)
            self._log_no_target("wait", "alive_or_pending_tracks")
            return False

        return self._handle_no_targets_impl()

    @abstractmethod
    def _handle_no_targets_impl(self) -> bool:
        """Mode-specific no-target behaviour (teleport, walk-wait, …).

        Called by ``on_no_attackable_targets()`` after common guards pass.
        """
        ...

    # ── Shared helpers used by concrete strategies ───────────────

    def _can_consider_area_clear(self) -> bool:
        """Check preconditions for considering the area clear."""
        if self._ctx.vision_busy():
            self._log_no_target_blocked("vision_busy")
            return False
        if self._ctx.urgent.has_pending():
            self._log_no_target_blocked("direct_state_pending")
            return False
        return True

    def _build_no_target_context(self) -> dict[str, object]:
        ctx = self._ctx
        now = monotonic_ms()
        area = ctx.tracks.get_area_clear_candidate(now)
        return {
            "attackable_count": ctx.tracks.get_attackable_count(now),
            "alive_or_pending_count": ctx.tracks.get_alive_or_pending_count(now),
            "area_clear": area.clear,
            "vision_busy": ctx.vision_busy(),
            "direct_state_pending": ctx.urgent.has_pending(),
            "has_discovery_since_reset": self._discovery_since_reset,
        }

    def _log_no_target(
        self,
        decision: str,
        reason: str,
        context: dict | None = None,
    ) -> None:
        ctx = self._ctx
        ctx_data = context or self._build_no_target_context()
        ctx.validation.log_no_target_decision(
            decision,
            reason,
            attackable_count=int(ctx_data["attackable_count"]),
            alive_or_pending_count=int(ctx_data["alive_or_pending_count"]),
            area_clear=bool(ctx_data["area_clear"]),
            vision_busy=bool(ctx_data["vision_busy"]),
            direct_state_pending=bool(ctx_data["direct_state_pending"]),
            has_discovery_since_reset=bool(
                ctx_data["has_discovery_since_reset"]
            ),
        )

    def _log_no_target_blocked(self, reason: str) -> None:
        now = monotonic_ms()
        if now - self._last_no_target_blocked_log_ms < 2000:
            return
        self._last_no_target_blocked_log_ms = now
        self._ctx.logger.behavior(f"[MODE] no-target blocked reason={reason}")

    def _log_wait_known(self, alive_or_pending_count: int) -> None:
        now = monotonic_ms()
        if now - self._last_wait_known_log_ms < 2000:
            return
        self._last_wait_known_log_ms = now
        self._ctx.logger.behavior(
            f"[MODE] wait reason=known_tracks_not_attackable "
            f"aliveOrPending={alive_or_pending_count}"
        )


class TeleportStrategy(HuntModeStrategy):
    """Teleport when area is clear of mobs."""

    def _handle_no_targets_impl(self) -> bool:
        ctx = self._ctx
        context = self._build_no_target_context()

        if not self._can_consider_area_clear():
            self._log_no_target("wait", "area_not_clearable", context)
            return False
        if not self._discovery_since_reset:
            self._log_no_target_blocked("no_discovery_yet")
            self._log_no_target("wait", "no_discovery_yet", context)
            return False

        area = ctx.tracks.get_area_clear_candidate()
        if not area.clear:
            self._log_no_target_blocked(area.reason)
            self._log_no_target("wait", area.reason, context)
            return False

        if not ctx.config.teleport_scan_code:
            self._log_no_target("wait", "no_teleport_key", context)
            return False

        ctx.logger.behavior(
            f"[MODE] teleport area_clear tracks={area.alive_or_pending_count}"
        )
        self._log_no_target("teleport", "area_clear", context)

        self._input.teleport_key(ctx.config.teleport_scan_code)
        hunt_overlay.increment_teleports()
        time.sleep(ctx.config.teleport_duration_ms / 1000.0)

        ctx.area_reset("post_teleport")
        self.on_area_reset()
        ctx.discovery_wake.set()
        return True


class WalkStrategy(HuntModeStrategy):
    """Wait for mobs to path into detection range (no teleport)."""

    def __init__(
        self,
        ctx: HuntModeControllerContext,
        input_backend: InputBackend,
    ) -> None:
        super().__init__(ctx, input_backend)
        self._walk_idle_start_ms = 0

    def on_area_reset(self) -> None:
        super().on_area_reset()
        self._walk_idle_start_ms = 0

    def _handle_no_targets_impl(self) -> bool:
        ctx = self._ctx
        now = monotonic_ms()

        # Start the idle timer on first call
        if not self._walk_idle_start_ms:
            self._walk_idle_start_ms = now
            ctx.logger.behavior(
                "[MODE] walk mode — waiting for mobs to appear"
            )
            self._log_no_target(
                "wait", "walk_waiting", self._build_no_target_context()
            )
            return False

        # If we haven't had any discovery since area reset, just wait
        if not self._discovery_since_reset:
            idle_seconds = (now - self._walk_idle_start_ms) // 1000
            if idle_seconds > 0 and idle_seconds % 15 == 0:
                ctx.logger.behavior(
                    f"[MODE] walk waiting for first discovery elapsed={idle_seconds}s"
                )
            self._log_no_target("wait", "walk_no_discovery_yet")
            return False

        # Reset idle timer when we do have discovery
        self._walk_idle_start_ms = 0

        area = ctx.tracks.get_area_clear_candidate(now)
        if area.clear:
            ctx.logger.behavior("[MODE] area clear — waiting for respawns")
            self._log_no_target("wait", "walk_area_clear")
            return False

        # We have pending tracks, just wait
        self._log_no_target("wait", "walk_has_pending")
        return False
