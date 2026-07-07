"""Shared primitive capability protocols (Interface Segregation Principle).

These protocols define narrow interfaces for common runtime capabilities.
They are satisfied structurally by HuntRuntimeContext and can be composed
into more specific context protocols (see workers/_protocols.py).

Use these when you need to express that a component only requires a
subset of the full runtime context.
"""

from __future__ import annotations

import threading
from typing import Protocol

from pybot.runtime.capture.hunt_capture import HuntWindowCapture
from pybot.runtime.config import HuntRuntimeConfig
from pybot.runtime.hunt_policy import HuntPolicy
from pybot.runtime.hunt_tracks import HuntTracks
from pybot.runtime.logging import HuntLogger
from pybot.runtime.validation_log import HuntValidationLogger
from pybot.runtime.detection.detector_session import DetectorSession


class CanStop(Protocol):
    """Worker lifecycle control — stop/pause events and queries."""
    @property
    def stop_event(self) -> threading.Event: ...
    @property
    def pause_event(self) -> threading.Event: ...
    def is_stopped(self) -> bool: ...
    def should_run_workers(self) -> bool: ...


class CanLog(Protocol):
    """Logging capability — behavioral log output."""
    @property
    def logger(self) -> HuntLogger: ...


class HasConfig(Protocol):
    """Read-only configuration access."""
    @property
    def config(self) -> HuntRuntimeConfig: ...


class CanCapture(Protocol):
    """Screen capture (window ROI) capability."""
    @property
    def capture(self) -> HuntWindowCapture: ...


class CanDetect(Protocol):
    """Mob detection (discovery) capability."""
    @property
    def detector(self) -> DetectorSession: ...


class CanTrackLocal(Protocol):
    """Local track-following capability — a detector dedicated to tracking so it
    never contends with the discovery scan's lock."""
    @property
    def tracker(self) -> DetectorSession: ...


class CanTrack(Protocol):
    """Track management capability."""
    @property
    def tracks(self) -> HuntTracks: ...


class CanValidate(Protocol):
    """Validation logging capability."""
    @property
    def validation(self) -> HuntValidationLogger: ...


class CanPolicy(Protocol):
    """Attack-policy capability."""
    @property
    def policy(self) -> HuntPolicy: ...


class CanAreaReset(Protocol):
    """Area reset capability."""
    def area_reset(self, reason: str = "area_reset") -> None: ...


class CanWakeDiscovery(Protocol):
    """Discovery-wake signalling event."""
    @property
    def discovery_wake(self) -> threading.Event: ...
