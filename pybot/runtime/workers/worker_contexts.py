"""Focused context protocols for hunt runtime workers (Interface Segregation Principle).

Primitive capability protocols are defined in pybot._protocols and re-exported
here for convenience. This module adds worker-specific combined protocols
that compose those primitives into narrow interfaces for each worker.

HuntRuntimeContext structurally satisfies all of them, but no worker
depends on the full god object.

Pause matrix (see ``runtime_context`` module docstring):
  sit     → discovery, tracking, attack, timers idle
  storage → discovery, tracking, attack idle; timers keep running
  heal    → attack, timers idle; discovery/tracking keep running
  sit ↔ storage ↔ heal mutually exclusive
"""

from __future__ import annotations

from typing import Protocol

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
    def wait_while_combat_blocked(self, timeout_s: float) -> bool: ...
    def character_screen_pos(self) -> tuple[int, int] | None: ...


class SkillTimerWorkerContext(CanStop, CanLog, HasConfig, Protocol):
    """Hunt runtime subset consumed by SkillTimerWorker.

    Uses ``should_run_timers``: idle during sit/pause/heal; keep firing
    during storage so timer schedules are not re-armed mid-session.
    """

    @property
    def resume_gate(self) -> object: ...

    def should_run_timers(self) -> bool: ...


class HpRestoreWorkerContext(CanStop, CanLog, HasConfig, CanCapture, Protocol):
    """Hunt runtime subset consumed by HpRestoreWorker.

    Idle during sit/pause (``should_run_workers``); keeps running during
    storage. Heal-until-full pauses combat via ``begin_heal_ops`` when HP
    is not full and safe, or during the post-teleport heal window. Danger TP
    and pause abort an in-progress heal-until-full session.
    """

    @property
    def discovery_suspend(self) -> object: ...
    @property
    def pause_event(self) -> object: ...

    def character_screen_pos(self) -> tuple[int, int] | None: ...
    def begin_heal_ops(self) -> bool: ...
    def end_heal_ops(self) -> None: ...
    def in_post_teleport_heal_window(self) -> bool: ...
    def wait_unless_stopped(self, timeout_s: float) -> bool: ...
    def wait_unless_paused_or_suspended(self, timeout_s: float) -> bool: ...


class CharacterStateWorkerContext(
    CanStop,
    CanLog,
    CanCapture,
    CanTrack,
    Protocol,
):
    """Hunt runtime subset consumed by CharacterStateMonitor.

    Runs detection while workers are running (``should_run_workers``).
    """
    pass


class SitOnLowSpWorkerContext(
    CanStop,
    CanLog,
    HasConfig,
    CanCapture,
    CanDetect,
    CanOverlay,
    CanWakeDiscovery,
    CanAreaReset,
    Protocol,
):
    """Hunt runtime subset consumed by SitOnLowSpWorker."""

    @property
    def discovery_wake(self) -> object: ...

    def begin_sit_ops(self) -> bool: ...
    def end_sit_ops(self) -> None: ...
    def wait_unless_stopped(self, timeout_s: float) -> bool: ...


class ItemsToStorageWorkerContext(
    CanStop,
    CanLog,
    HasConfig,
    CanCapture,
    CanDetect,
    CanOverlay,
    CanWakeDiscovery,
    CanAreaReset,
    Protocol,
):
    """Hunt runtime subset consumed by ItemsToStorageWorker."""

    wingcount: int
    fly_wings_exhausted: bool
    sitting_event: object

    def begin_storage_ops(self) -> bool: ...
    def end_storage_ops(self) -> None: ...
    def wait_unless_stopped(self, timeout_s: float) -> bool: ...
    def should_restock_fly_wings(self) -> bool: ...
    def mark_fly_wings_exhausted(self) -> None: ...
