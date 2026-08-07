"""Focused context protocols for hunt runtime workers (Interface Segregation Principle).

Primitive capability protocols are defined in pybot._protocols and re-exported
here for convenience. This module adds worker-specific combined protocols
that compose those primitives into narrow interfaces for each worker.

HuntRuntimeContext structurally satisfies all of them, but no worker
depends on the full god object.

Pause matrix (see ``runtime_context`` module docstring):
  sit     → discovery/tracking keep sampling; attack/timers idle
  storage → discovery/tracking keep sampling; attack idle; timers keep running
  heal    → discovery/tracking/timers keep running; attack idle
  sit ↔ storage ↔ heal mutually exclusive
"""

from __future__ import annotations

from typing import Protocol

from pybot.runtime.session_lifecycle import SessionLifecycle
from pybot._protocols import (
    CanAreaReset,
    CanCapture,
    CanDetect,
    CanLog,
    CanOverlay,
    CanPolicy,
    CanStop,
    CanTrack,
    CanTrackLocal,
    CanValidate,
    CanWakeDiscovery,
    HasConfig,
)

# ── HuntModeController context protocol ──────────────────────────


class HuntModeControllerContext(
    CanStop,
    CanLog,
    HasConfig,
    CanTrack,
    CanValidate,
    CanPolicy,
    CanWakeDiscovery,
    CanAreaReset,
    CanOverlay,
    Protocol,
):
    """Hunt runtime subset consumed by HuntModeController / strategies."""

    fly_wings_exhausted: bool

    def should_run_combat(self) -> bool: ...
    def should_run_mode_transitions(self) -> bool: ...
    def perform_input_if_allowed(self, allowed, action) -> bool: ...
    def wait_unless_stopped(self, timeout_s: float) -> bool: ...


# ── Worker-specific combined context protocols ────────────────────
# Each lists exactly what its worker touches from the runtime context.


class CoordTrackingWorkerContext(
    CanStop,
    CanLog,
    CanCapture,
    CanTrackLocal,
    CanTrack,
    CanWakeDiscovery,
    CanOverlay,
    Protocol,
):
    """Hunt runtime subset consumed by CoordTrackingWorker."""

    @property
    def resume_gate(self) -> object: ...

    def should_run_tracking(self) -> bool: ...


class DiscoveryWorkerContext(
    CanStop,
    CanLog,
    HasConfig,
    CanCapture,
    CanDetect,
    CanTrack,
    CanValidate,
    CanWakeDiscovery,
    CanOverlay,
    Protocol,
):
    """Hunt runtime subset consumed by DiscoveryWorker."""

    def should_run_discovery(self) -> bool: ...


class AttackLoopContext(
    CanStop,
    CanLog,
    HasConfig,
    CanTrack,
    CanValidate,
    CanPolicy,
    CanOverlay,
    Protocol,
):
    """Hunt runtime subset consumed by AttackLoop."""

    def should_run_combat(self) -> bool: ...
    def should_run_mode_transitions(self) -> bool: ...
    def should_run_custom_heal_actions(self) -> bool: ...
    def try_heal_if_allowed(self, allowed, action) -> str: ...
    def in_post_teleport_heal_window(self) -> bool: ...
    def clear_post_teleport_heal(self) -> None: ...
    def wait_while_combat_blocked(self, timeout_s: float) -> bool: ...
    def character_screen_pos(self) -> tuple[int, int] | None: ...


class SelfBuffWorkerContext(CanStop, CanLog, HasConfig, Protocol):
    """Hunt runtime subset consumed by per-mob self-buff casts."""

    @property
    def character_action_gate(self) -> object: ...

    def should_run_combat(self) -> bool: ...
    def should_run_character_actions(self) -> bool: ...
    def should_run_startup_actions(self) -> bool: ...
    def mark_startup_buffs_done(
        self,
        *,
        expected_generation: int | None = None,
    ) -> bool: ...
    def mark_startup_timers_done(
        self,
        *,
        expected_generation: int | None = None,
    ) -> bool: ...
    def wait_while_combat_blocked(self, timeout_s: float) -> bool: ...
    def character_screen_pos(self) -> tuple[int, int] | None: ...

    @property
    def hunt_generation(self) -> int: ...

    @property
    def startup_timers_done(self) -> object: ...


class SkillTimerWorkerContext(CanStop, CanLog, HasConfig, Protocol):
    """Hunt runtime subset consumed by SkillTimerWorker.

    Uses ``should_run_timers``: idle during sit/pause; keep firing during
    storage/healing so timer schedules are not re-armed mid-session.
    """

    @property
    def resume_gate(self) -> object: ...

    @property
    def character_action_gate(self) -> object: ...

    def should_run_timers(self) -> bool: ...
    def should_run_startup_actions(self) -> bool: ...
    def mark_startup_buffs_done(
        self,
        *,
        expected_generation: int | None = None,
    ) -> bool: ...

    @property
    def startup_buffs_done(self) -> object: ...

    @property
    def hunt_generation(self) -> int: ...

    @property
    def startup_timers_done(self) -> object: ...

    def mark_startup_timers_done(
        self,
        *,
        expected_generation: int | None = None,
    ) -> bool: ...


class HpRestoreWorkerContext(CanStop, CanLog, HasConfig, Protocol):
    """Hunt runtime subset consumed by the HP item worker."""

    def should_run_workers(self) -> bool: ...
    def wait_while_stopped_or_paused(self, timeout_s: float) -> bool: ...


class SitOnLowSpWorkerContext(
    CanStop,
    CanLog,
    HasConfig,
    CanWakeDiscovery,
    SessionLifecycle,
    Protocol,
):
    """Hunt runtime subset consumed by SitOnLowSpWorker.

    Capture/detect for area-clear live on ``TeleportController``'s ctx, not
    here. Pose OCR is unused — sit/stand is press-once with a seated flag.
    """

    @property
    def discovery_wake(self) -> object: ...
    @property
    def pause_event(self) -> object: ...

    @property
    def danger_sit_requested(self) -> object: ...
    @property
    def critical_danger_requested(self) -> object: ...
    @property
    def danger_escape_active(self) -> object: ...

    def should_run_workers(self) -> bool: ...
    def request_danger_sit(self) -> bool: ...
    def begin_sit_ops(self) -> bool: ...
    def try_begin_sit_ops(self) -> bool: ...
    def pop_danger_sit_request(self) -> bool: ...
    def begin_danger_escape(self) -> bool: ...
    def try_begin_critical_escape_ops(self, *, override: bool = False) -> bool: ...
    def wait_for_preempted_session_release(self, timeout_s: float) -> bool: ...
    def preempted_sessions(self) -> tuple[bool, bool, bool]: ...
    def end_danger_escape(self) -> None: ...
    def end_critical_escape_ops(self) -> None: ...
    def end_sit_ops(self, *, trusted_clear: bool = True) -> None: ...
    def wait_unless_stopped(self, timeout_s: float) -> bool: ...
    def wait_while_user_paused(self, timeout_s: float) -> bool: ...


class ItemsToStorageWorkerContext(
    CanStop,
    CanLog,
    HasConfig,
    CanCapture,
    CanDetect,
    CanOverlay,
    CanWakeDiscovery,
    CanAreaReset,
    SessionLifecycle,
    Protocol,
):
    """Hunt runtime subset consumed by ItemsToStorageWorker."""

    wingcount: int
    fly_wings_exhausted: bool
    sitting_event: object
    healing_event: object
    danger_sit_requested: object
    critical_danger_requested: object
    danger_escape_active: object
    critical_danger_escape_active: object

    def storage_due(self) -> bool: ...
    def can_execute_now(self) -> bool: ...
    def begin_storage_ops(self) -> bool: ...
    def end_storage_ops(self) -> None: ...
    def wait_unless_stopped(self, timeout_s: float) -> bool: ...
    def should_restock_fly_wings(self) -> bool: ...
    def mark_fly_wings_exhausted(self) -> None: ...
