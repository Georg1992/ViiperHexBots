"""Shared hunt runtime state for all workers.

This dataclass holds references to all runtime services. Event gate logic,
wing management, and other behaviours are delegated to specialised classes
(:class:`GateController`, :class:`WingTracker`) so the dataclass does not
accumulate unrelated business logic.

Pause / session matrix
----------------------
Who may run while a signal is held:

======= ========= =========== ====== ======
Signal  Discovery Tracking    Attack Timers
======= ========= =========== ====== ======
(none)  yes       yes         yes    yes
pause   no        no          no     no
sit     no        no          no     no
storage no        no          no     yes
heal    yes       yes         no     no
======= ========= =========== ====== ======

Sit, storage, and heal are mutually exclusive.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from pybot.runtime.capture.hunt_capture import HuntWindowCapture
from pybot.runtime.config import HuntRuntimeConfig
from pybot.runtime.control import RuntimeControl
from pybot.runtime.gate_controller import GateController
from pybot.runtime.hunt_policy import HuntPolicy
from pybot.runtime.hunt_tracks import HuntTracks
from pybot.runtime.logging import HuntLogger
from pybot.runtime.validation_log import HuntValidationLogger
from pybot.runtime.detection.detector_session import DetectorSession
from pybot.runtime.overlay_ports import HuntOverlay, NullOverlay
from pybot.runtime.wing_tracker import WingTracker


@dataclass
class HuntRuntimeContext:
    """Shared runtime state and service references for all workers.

    Event gate logic (should_run_*, begin/end sit/storage/heal, wait helpers)
    is delegated to :attr:`gates`.
    Wing counter management is delegated to :attr:`wings`.
    """
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
    # Event gate + worker lifecycle (not a field — always created fresh).
    gates: GateController = field(default_factory=GateController)
    # Fly-wing count and restock state.
    wings: WingTracker = field(default_factory=WingTracker)

    # ── Event gate convenience properties (delegate to gates) ────

    @property
    def stop_event(self) -> threading.Event:
        return self.gates.stop_event

    @stop_event.setter
    def stop_event(self, event: threading.Event) -> None:
        self.gates.stop_event = event

    @property
    def pause_event(self) -> threading.Event:
        return self.gates.pause_event

    @pause_event.setter
    def pause_event(self, event: threading.Event) -> None:
        self.gates.pause_event = event

    @property
    def resume_gate(self) -> threading.Event:
        return self.gates.resume_gate

    @resume_gate.setter
    def resume_gate(self, event: threading.Event) -> None:
        self.gates.resume_gate = event

    @property
    def discovery_wake(self) -> threading.Event:
        return self.gates.discovery_wake

    @discovery_wake.setter
    def discovery_wake(self, event: threading.Event) -> None:
        self.gates.discovery_wake = event

    @property
    def discovery_suspend(self) -> threading.Event:
        return self.gates.discovery_suspend

    @discovery_suspend.setter
    def discovery_suspend(self, event: threading.Event) -> None:
        self.gates.discovery_suspend = event

    @property
    def sitting_event(self) -> threading.Event:
        return self.gates.sitting_event

    @sitting_event.setter
    def sitting_event(self, event: threading.Event) -> None:
        self.gates.sitting_event = event

    @property
    def storage_event(self) -> threading.Event:
        return self.gates.storage_event

    @storage_event.setter
    def storage_event(self, event: threading.Event) -> None:
        self.gates.storage_event = event

    @property
    def healing_event(self) -> threading.Event:
        return self.gates.healing_event

    @healing_event.setter
    def healing_event(self, event: threading.Event) -> None:
        self.gates.healing_event = event

    # ── Wing convenience properties (delegate to wings) ──────────

    @property
    def wingcount(self) -> int:
        return self.wings.wingcount

    @wingcount.setter
    def wingcount(self, value: int) -> None:
        self.wings.wingcount = value

    @property
    def fly_wings_exhausted(self) -> bool:
        return self.wings.fly_wings_exhausted

    @fly_wings_exhausted.setter
    def fly_wings_exhausted(self, value: bool) -> None:
        self.wings.fly_wings_exhausted = value

    # ── Gate convenience methods (delegate to gates) ─────────────

    def should_run_workers(self) -> bool:
        return self.gates.should_run_workers()

    def should_run_combat(self) -> bool:
        return self.gates.should_run_combat()

    def should_run_timers(self) -> bool:
        return self.gates.should_run_timers()

    def should_allow_danger_teleport(self) -> bool:
        return self.gates.should_allow_danger_teleport()

    def should_run_discovery(self) -> bool:
        return self.gates.should_run_discovery()

    def should_run_tracking(self) -> bool:
        return self.gates.should_run_tracking()

    def is_stopped(self) -> bool:
        return self.gates.is_stopped()

    def mark_running(self) -> None:
        self.gates.mark_running()

    def mark_paused(self) -> None:
        self.gates.mark_paused()

    def try_begin_sit_ops(self) -> bool:
        return self.gates.try_begin_sit_ops()

    def begin_sit_ops(self) -> bool:
        return self.gates.begin_sit_ops()

    def end_sit_ops(self) -> None:
        self.gates.end_sit_ops()

    def try_begin_storage_ops(self) -> bool:
        return self.gates.try_begin_storage_ops()

    def begin_storage_ops(self) -> bool:
        return self.gates.begin_storage_ops()

    def end_storage_ops(self) -> None:
        self.gates.end_storage_ops()

    def try_begin_heal_ops(self) -> bool:
        return self.gates.try_begin_heal_ops()

    def begin_heal_ops(self) -> bool:
        return self.gates.begin_heal_ops()

    def end_heal_ops(self) -> None:
        self.gates.end_heal_ops()

    def mark_post_teleport_heal(self, duration_s: float) -> None:
        self.gates.mark_post_teleport_heal(duration_s)

    def in_post_teleport_heal_window(self) -> bool:
        return self.gates.in_post_teleport_heal_window()

    def wait_while_stopped_or_paused(self, timeout_s: float) -> bool:
        return self.gates.wait_while_stopped_or_paused(timeout_s)

    def wait_while_user_paused(self, timeout_s: float) -> bool:
        return self.gates.wait_while_user_paused(timeout_s)

    def wait_while_combat_blocked(self, timeout_s: float) -> bool:
        return self.gates.wait_while_combat_blocked(timeout_s)

    def wait_unless_stopped(self, timeout_s: float) -> bool:
        return self.gates.wait_unless_stopped(timeout_s)

    def wait_unless_paused_or_suspended(self, timeout_s: float) -> bool:
        return self.gates.wait_unless_paused_or_suspended(timeout_s)

    # ── Wing convenience methods (delegate to wings) ─────────────

    def note_teleport_for_wings(self) -> None:
        self.wings.note_teleport(
            open_storage_steps=bool(self.config.open_storage_steps),
            take_fly_wings=self.config.take_fly_wings,
        )

    def should_restock_fly_wings(self) -> bool:
        return self.wings.should_restock(
            open_storage_steps=bool(self.config.open_storage_steps),
            take_fly_wings=self.config.take_fly_wings,
            fly_wings_amount=int(self.config.fly_wings_amount),
        )

    def mark_fly_wings_exhausted(self) -> None:
        self.wings.mark_exhausted()

    # ── Own methods (not delegated) ──────────────────────────────

    def character_screen_pos(self) -> tuple[int, int] | None:
        """Hunt ROI center — character is always at the middle of the hunt view."""
        roi = self.capture.get_hunt_roi()
        if roi is None:
            return None
        return roi.x + roi.w // 2, roi.y + roi.h // 2

    def area_reset(self, reason: str = "area_reset") -> None:
        self.tracks.area_reset()
        self.policy.reset()
        self.validation.log_area_reset(reason)
