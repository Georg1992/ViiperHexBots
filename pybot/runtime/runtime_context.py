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
    # Set while regenerating SP (sit) — hunt + skill timers idle.
    sitting_event: threading.Event = field(default_factory=threading.Event)
    # Set while ItemsToStorage / GetFlyWings runs — combat idles; timers keep going.
    storage_event: threading.Event = field(default_factory=threading.Event)
    _exclusive_lock: threading.Lock = field(default_factory=threading.Lock)
    # AHK ``wingcount``: remaining fly wings; restocked by GetFlyWings.
    wingcount: int = 0
    # Set when storage has no wings left — stop GetFlyWings for this hunt.
    fly_wings_exhausted: bool = False

    def should_run_workers(self) -> bool:
        """True when hunt + skill timers may run (not stopped/paused/sitting)."""
        return (
            not self.stop_event.is_set()
            and not self.pause_event.is_set()
            and not self.sitting_event.is_set()
        )

    def should_run_combat(self) -> bool:
        """True when attack may run (workers + not in a storage session)."""
        return self.should_run_workers() and not self.storage_event.is_set()

    def mark_running(self) -> None:
        """Workers may run; wake any thread blocked in ``wait_while_stopped_or_paused``."""
        self.pause_event.clear()
        if not self.sitting_event.is_set():
            self.resume_gate.set()

    def mark_paused(self) -> None:
        """Workers must idle until ``mark_running``."""
        self.pause_event.set()
        self.resume_gate.clear()

    def try_begin_exclusive_ops(self) -> bool:
        """Acquire sit pause (hunt + timers). False if sit or storage already held."""
        with self._exclusive_lock:
            if self.sitting_event.is_set() or self.storage_event.is_set():
                return False
            self.sitting_event.set()
            self.resume_gate.clear()
            return True

    def begin_exclusive_ops(self) -> bool:
        """Wait until sit exclusive ops can start. False if stopped first."""
        while not self.stop_event.is_set():
            if self.try_begin_exclusive_ops():
                return True
            self.stop_event.wait(WORKER_POLL_INTERVAL_S)
        return False

    def end_exclusive_ops(self) -> None:
        """Release sit pause."""
        with self._exclusive_lock:
            self.sitting_event.clear()
            if not self.pause_event.is_set() and not self.stop_event.is_set():
                self.resume_gate.set()

    def try_begin_storage_ops(self) -> bool:
        """Acquire storage session (combat only). False if sit/storage held."""
        with self._exclusive_lock:
            if self.sitting_event.is_set() or self.storage_event.is_set():
                return False
            self.storage_event.set()
            return True

    def begin_storage_ops(self) -> bool:
        """Wait until storage can start. False if stopped first."""
        while not self.stop_event.is_set():
            if self.try_begin_storage_ops():
                return True
            self.stop_event.wait(WORKER_POLL_INTERVAL_S)
        return False

    def end_storage_ops(self) -> None:
        """Release storage session; combat may resume."""
        with self._exclusive_lock:
            self.storage_event.clear()

    def begin_sit_regen(self) -> None:
        """Pause hunting/timers for SP regeneration (independent of user pause)."""
        self.begin_exclusive_ops()

    def end_sit_regen(self) -> None:
        """Resume hunting/timers after sit regen completes."""
        self.end_exclusive_ops()

    def note_teleport_for_wings(self) -> None:
        """AHK Teleport: decrement wing counter when Take Fly Wings is on."""
        if (
            self.config.open_storage_steps
            and self.config.take_fly_wings
            and not self.fly_wings_exhausted
            and self.wingcount > 0
        ):
            self.wingcount -= 1

    def should_restock_fly_wings(self) -> bool:
        """True when GetFlyWings should run (enabled, not exhausted, count 0)."""
        return (
            bool(self.config.open_storage_steps)
            and self.config.take_fly_wings
            and not self.fly_wings_exhausted
            and self.wingcount <= 0
        )

    def mark_fly_wings_exhausted(self) -> None:
        """Stop fly-wing restock for this hunt; teleports switch to Creamy TP."""
        self.fly_wings_exhausted = True
        self.wingcount = 0

    def active_teleport_scan_code(self) -> int:
        """Fly-wing teleport, or Creamy TP when wings are exhausted / Creamy mob."""
        if self.fly_wings_exhausted:
            return self.config.creamy_tp_scan_code
        return self.config.active_teleport_scan_code()

    def active_teleport_button(self) -> str:
        if self.fly_wings_exhausted:
            return self.config.creamy_tp_button
        return self.config.active_teleport_button()

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

    def wait_while_combat_blocked(self, timeout_s: float) -> bool:
        """Block while sit/pause/storage holds combat. True if combat may run."""
        deadline = time.monotonic() + timeout_s
        while not self.stop_event.is_set():
            if self.should_run_combat():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self.should_run_combat()
            # Storage does not clear resume_gate; poll stop/sit wake.
            if self.sitting_event.is_set() or self.pause_event.is_set():
                self.resume_gate.wait(min(WORKER_POLL_INTERVAL_S, remaining))
            else:
                self.stop_event.wait(min(WORKER_POLL_INTERVAL_S, remaining))
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
