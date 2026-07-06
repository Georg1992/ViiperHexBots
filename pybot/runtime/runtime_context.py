"""Shared hunt runtime state for all workers."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from pybot.runtime.capture.hunt_capture import HuntWindowCapture
from pybot.runtime.config import HuntRuntimeConfig
from pybot.runtime.control import RuntimeControl
from pybot.runtime.hunt_policy import HuntPolicy
from pybot.runtime.hunt_tracks import HuntTracks
from pybot.runtime.logging import HuntLogger
from pybot.runtime.urgent_state import UrgentStateQueue
from pybot.runtime.validation_log import HuntValidationLogger
from pybot.runtime.detection.detector_session import DetectorSession


@dataclass
class HuntRuntimeContext:
    config: HuntRuntimeConfig
    logger: HuntLogger
    tracks: HuntTracks
    policy: HuntPolicy
    capture: HuntWindowCapture
    detector: DetectorSession
    urgent: UrgentStateQueue
    validation: HuntValidationLogger
    control: RuntimeControl
    stop_event: threading.Event = field(default_factory=threading.Event)
    pause_event: threading.Event = field(default_factory=threading.Event)
    discovery_wake: threading.Event = field(default_factory=threading.Event)

    def should_run_workers(self) -> bool:
        return not self.stop_event.is_set() and not self.pause_event.is_set()

    def is_stopped(self) -> bool:
        return self.stop_event.is_set()

    def area_reset(self, reason: str = "area_reset") -> None:
        self.urgent.clear()
        self.tracks.area_reset()
        self.policy.reset()
        self.validation.log_area_reset(reason)

    def vision_busy(self) -> bool:
        return self.detector.is_busy()
