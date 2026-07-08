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
                                CanTrack, CanValidate,
                                CanWakeDiscovery,
                                CanAreaReset, CanOverlay, Protocol):
    """Hunt runtime subset consumed by HuntModeController."""
    pass


# ── Worker-specific combined context protocols ────────────────────
# Each lists exactly what its worker touches from the runtime context.


class TrackingWorkerContext(CanStop, CanLog,
                            CanCapture, CanTrackLocal, CanTrack,
                            CanOverlay, Protocol):
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
