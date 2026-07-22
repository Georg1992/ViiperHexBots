"""Focused context protocols for hunt runtime workers (Interface Segregation Principle).

Primitive capability protocols are defined in pybot._protocols and re-exported
here for convenience. This module adds worker-specific combined protocols
that compose those primitives into narrow interfaces for each worker.

HuntRuntimeContext structurally satisfies all of them, but no worker
depends on the full god object.
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


class HuntModeControllerContext(CanStop, CanLog, HasConfig,
                                CanTrack, CanValidate, CanPolicy,
                                CanWakeDiscovery,
                                CanAreaReset, CanOverlay, Protocol):
    """Hunt runtime subset consumed by HuntModeController."""
    pass


# ── Worker-specific combined context protocols ────────────────────
# Each lists exactly what its worker touches from the runtime context.


class TrackingWorkerContext(CanStop, CanLog,
                            CanCapture, CanTrackLocal, CanTrack,
                            CanWakeDiscovery, CanOverlay, Protocol):
    """Hunt runtime subset consumed by TrackingWorker."""
    pass


class DiscoveryWorkerContext(CanStop, CanLog, HasConfig,
                             CanCapture, CanDetect, CanTrack,
                             CanValidate, CanWakeDiscovery, CanOverlay,
                             Protocol):
    """Hunt runtime subset consumed by DiscoveryWorker."""
    pass


class AttackLoopContext(CanStop, CanLog, HasConfig,
                        CanTrack, CanValidate,
                        CanPolicy, CanOverlay, Protocol):
    """Hunt runtime subset consumed by AttackLoop."""
    pass


class SkillTimerWorkerContext(CanStop, CanLog, HasConfig, Protocol):
    """Hunt runtime subset consumed by SkillTimerWorker."""
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

    def begin_sit_regen(self) -> None: ...
    def end_sit_regen(self) -> None: ...
    def wait_unless_stopped(self, timeout_s: float) -> bool: ...


class ItemsToStorageWorkerContext(
    CanStop,
    CanLog,
    HasConfig,
    CanCapture,
    CanWakeDiscovery,
    Protocol,
):
    """Hunt runtime subset consumed by ItemsToStorageWorker."""

    wingcount: int
    sitting_event: object

    def begin_exclusive_ops(self) -> bool: ...
    def end_exclusive_ops(self) -> None: ...
    def wait_unless_stopped(self, timeout_s: float) -> bool: ...
