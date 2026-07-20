"""Shared hunt runtime state for all workers."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from pybot.runtime.capture.hunt_capture import HuntWindowCapture
from pybot.runtime.config import HuntRuntimeConfig
from pybot.runtime.control import RuntimeControl
from pybot.runtime.hunt_policy import HuntPolicy
from pybot.runtime.hunt_tracks import HuntTracks
from pybot.runtime.logging import HuntLogger
from pybot.runtime.validation_log import HuntValidationLogger
from pybot.runtime.detection.detector_session import DetectorSession
from pybot.runtime.constants import WORKER_POLL_INTERVAL_S
from pybot.runtime.overlay_ports import HuntOverlay, NullOverlay


@dataclass
class HuntRuntimeContext:
    config: HuntRuntimeConfig
    logger: HuntLogger
    tracks: HuntTracks
    policy: HuntPolicy
    capture: HuntWindowCapture
    detector: DetectorSession
    tracker: DetectorSession
    validation: HuntValidationLogger
    control: RuntimeControl
    overlay: HuntOverlay = field(default_factory=NullOverlay)
    stop_event: threading.Event = field(default_factory=threading.Event)
    pause_event: threading.Event = field(default_factory=threading.Event)
    resume_gate: threading.Event = field(default_factory=threading.Event)
    discovery_wake: threading.Event = field(default_factory=threading.Event)
    # Set for the whole claim → teleport key → settle delay window so the
    # 1s discovery cadence cannot scan mid-teleport and falsely confirm clear.
    discovery_suspend: threading.Event = field(default_factory=threading.Event)
    # Set while regenerating SP (sit) — hunting + skill timers idle until clear.
    sitting_event: threading.Event = field(default_factory=threading.Event)

    def should_run_workers(self) -> bool:
        return (
            not self.stop_event.is_set()
            and not self.pause_event.is_set()
            and not self.sitting_event.is_set()
        )

    def mark_running(self) -> None:
        """Workers may run; wake any thread blocked in ``wait_while_stopped_or_paused``."""
        self.pause_event.clear()
        if not self.sitting_event.is_set():
            self.resume_gate.set()

    def mark_paused(self) -> None:
        """Workers must idle until ``mark_running``."""
        self.pause_event.set()
        self.resume_gate.clear()

    def begin_sit_regen(self) -> None:
        """Pause hunting/timers for SP regeneration (independent of user pause)."""
        self.sitting_event.set()
        self.resume_gate.clear()

    def end_sit_regen(self) -> None:
        """Resume hunting/timers after sit regen completes."""
        self.sitting_event.clear()
        if not self.pause_event.is_set() and not self.stop_event.is_set():
            self.resume_gate.set()

    def is_stopped(self) -> bool:
        return self.stop_event.is_set()

    def wait_while_stopped_or_paused(self, timeout_s: float) -> bool:
        """Block up to *timeout_s*. Returns True if workers may run."""
        deadline = time.monotonic() + timeout_s
        while not self.stop_event.is_set():
            if self.should_run_workers():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self.should_run_workers()
            self.resume_gate.wait(min(WORKER_POLL_INTERVAL_S, remaining))
        return False

    def wait_unless_stopped(self, timeout_s: float) -> bool:
        """Wait up to *timeout_s* unless stop/pause is requested.

        Returns True only when the full duration elapsed without interruption.
        """
        deadline = time.monotonic() + timeout_s
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            if self.stop_event.wait(min(WORKER_POLL_INTERVAL_S, remaining)):
                return False
        return False

    def area_reset(self, reason: str = "area_reset") -> None:
        self.tracks.area_reset()
        self.policy.reset()
        self.validation.log_area_reset(reason)
