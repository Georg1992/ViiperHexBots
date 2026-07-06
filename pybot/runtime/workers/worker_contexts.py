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
    CanPolicy,
    CanStop,
    CanTrack,
    CanUrgentState,
    CanValidate,
    CanVisionBusy,
    CanWakeDiscovery,
    HasConfig,
)

# ── HuntModeController context protocol ──────────────────────────


class HuntModeControllerContext(CanStop, CanLog, HasConfig,
                                CanTrack, CanValidate, CanUrgentState,
                                CanWakeDiscovery, CanVisionBusy,
                                CanAreaReset, Protocol):
    """Hunt runtime subset consumed by HuntModeController."""
    pass


# ── Worker-specific combined context protocols ────────────────────
# Each lists exactly what its worker touches from the runtime context.


class DiscoveryWorkerContext(CanStop, CanLog, HasConfig,
                             CanCapture, CanDetect, CanTrack,
                             CanValidate, CanWakeDiscovery, Protocol):
    """Hunt runtime subset consumed by DiscoveryWorker."""
    pass


class TrackingWorkerContext(CanStop, CanLog, HasConfig,
                            CanCapture, CanDetect, CanTrack,
                            CanValidate, Protocol):
    """Hunt runtime subset consumed by TrackingWorker."""
    pass


class ConfirmStateWorkerContext(CanStop, CanLog, HasConfig,
                                CanCapture, CanDetect, CanTrack,
                                CanValidate, CanUrgentState, Protocol):
    """Hunt runtime subset consumed by ConfirmStateWorker."""
    pass


class AttackLoopContext(CanStop, CanLog, HasConfig,
                        CanTrack, CanValidate,
                        CanUrgentState, CanPolicy, Protocol):
    """Hunt runtime subset consumed by AttackLoop."""
    pass


class SkillTimerWorkerContext(CanStop, CanLog, HasConfig, Protocol):
    """Hunt runtime subset consumed by SkillTimerWorker."""
    pass
