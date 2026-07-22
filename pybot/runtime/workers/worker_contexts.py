"""Focused context protocols for hunt runtime workers (Interface Segregation Principle).

Primitive capability protocols are defined in pybot._protocols and re-exported
here for convenience. This module adds worker-specific combined protocols
that compose those primitives into narrow interfaces for each worker.

HuntRuntimeContext structurally satisfies all of them, but no worker
depends on the full god object.

Pause matrix (see ``runtime_context`` module docstring):
  sit     → discovery, tracking, attack, timers idle
  storage → attack idle only; timers keep running
  sit ↔ storage mutually exclusive
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
    def active_teleport_scan_code(self) -> int: ...
    def active_teleport_button(self) -> str: ...
    def note_teleport_for_wings(self) -> None: ...


# ── Worker-specific combined context protocols ────────────────────
# Each lists exactly what its worker touches from the runtime context.


class TrackingWorkerContext(
    CanStop,
    CanLog,
    CanCapture,
    CanTrackLocal,
    CanTrack,
    CanWakeDiscovery,
    CanOverlay,
    Protocol,
):
    """Hunt runtime subset consumed by TrackingWorker."""
    pass


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
    pass


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


class SkillTimerWorkerContext(CanStop, CanLog, HasConfig, Protocol):
    """Hunt runtime subset consumed by SkillTimerWorker.

    Uses ``should_run_workers`` (via CanStop): idle during sit/pause, keep
    firing during storage so timer schedules are not re-armed mid-session.
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

    def begin_sit_regen(self) -> bool: ...
    def end_sit_regen(self) -> None: ...
    def wait_unless_stopped(self, timeout_s: float) -> bool: ...
    def active_teleport_scan_code(self) -> int: ...
    def active_teleport_button(self) -> str: ...
    def note_teleport_for_wings(self) -> None: ...


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
    def active_teleport_scan_code(self) -> int: ...
    def active_teleport_button(self) -> str: ...
    def note_teleport_for_wings(self) -> None: ...
