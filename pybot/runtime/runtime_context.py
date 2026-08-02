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
heal    yes       yes         no     yes
======= ========= =========== ====== ======

Sit, storage, and heal are mutually exclusive.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from pybot.runtime.capture.hunt_capture import HuntWindowCapture
from pybot.config.runtime import HuntRuntimeConfig
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
    # Shared danger observer used by safe character actions (heal/buff).
    danger_detector: object | None = field(default=None, repr=False)
    # Registered by the sit worker so runtime shutdown can retry an unresolved
    # seated toggle after the worker thread has exited.
    sit_cleanup_callback: Callable[[], bool] | None = field(
        default=None, repr=False,
    )

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

    @property
    def danger_sit_requested(self) -> threading.Event:
        """Pending seated-damage request raised by DangerDetector."""
        return self.gates.danger_sit_requested

    @property
    def critical_danger_requested(self) -> threading.Event:
        """Pending critical hunting escape request."""
        return self.gates.critical_danger_requested

    @property
    def sit_cleanup_unresolved(self) -> threading.Event:
        """True when a seated toggle still needs shutdown cleanup."""
        return self.gates.sit_cleanup_unresolved

    def register_sit_cleanup(self, callback: Callable[[], bool]) -> None:
        """Register the sit worker's narrow shutdown cleanup operation."""
        self.sit_cleanup_callback = callback

    def mark_sit_cleanup_unresolved(self) -> None:
        self.gates.mark_sit_cleanup_unresolved()

    def clear_sit_cleanup_unresolved(self) -> None:
        self.gates.clear_sit_cleanup_unresolved()

    def retry_sit_cleanup(self) -> bool:
        """Retry unresolved seated cleanup without restarting worker threads."""
        if not self.sit_cleanup_unresolved.is_set():
            return True
        callback = self.sit_cleanup_callback
        if callback is None or not callback():
            return False
        self.clear_sit_cleanup_unresolved()
        return True

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
        """True when lifecycle gates and per-hunt startup both permit combat."""
        return (
            self.gates.should_run_combat()
            and self.gates.startup.is_combat_ready()
        )

    def perform_input_if_allowed(self, allowed, action) -> bool:
        """Admit one short input action against session transitions."""
        return self.gates.perform_input_if_allowed(allowed, action)

    def perform_heal_if_allowed(self, allowed, action, *, cooldown_s: float = 1.0) -> bool:
        """Admit healing through the shared cross-worker cooldown."""
        return self.gates.perform_heal_if_allowed(
            allowed, action, cooldown_s=cooldown_s
        )

    def should_run_timers(self) -> bool:
        return self.gates.should_run_timers()

    def should_run_mode_transitions(self) -> bool:
        """True when a teleport transition may claim the hunt input boundary."""
        return (
            self.should_run_combat()
            and self.startup_buffs_done.is_set()
            and self.startup_timers_done.is_set()
        )

    @property
    def hunt_generation(self) -> int:
        return self.gates.startup.generation

    @property
    def startup_area_clear(self) -> threading.Event:
        return self.gates.startup.area_clear

    @property
    def startup_buffs_done(self) -> threading.Event:
        return self.gates.startup.buffs_done

    @property
    def startup_timers_done(self) -> threading.Event:
        return self.gates.startup.timers_done

    def mark_startup_area_clear(
        self,
        clear: bool = True,
        *,
        expected_generation: int | None = None,
    ) -> bool:
        return self.gates.startup.mark_area_clear(
            clear,
            expected_generation=expected_generation,
        )

    def mark_startup_buffs_done(
        self,
        *,
        expected_generation: int | None = None,
    ) -> bool:
        return self.gates.startup.mark_buffs_done(
            expected_generation=expected_generation,
        )

    def mark_startup_timers_done(
        self,
        *,
        expected_generation: int | None = None,
    ) -> bool:
        return self.gates.startup.mark_timers_done(
            expected_generation=expected_generation,
        )

    def should_run_character_actions(self) -> bool:
        """True when a self-targeted action may run without active danger."""
        if not self.should_run_combat():
            return False
        danger = self.danger_detector
        if danger is None:
            return True
        return bool(danger.is_safe_for_heal())

    def should_run_heal_actions(self) -> bool:
        """True when healing is allowed in the post-teleport safety window."""
        return (
            self.should_run_character_actions()
            and self.in_post_teleport_heal_window()
        )

    def should_run_startup_actions(self) -> bool:
        """True when a new-hunt startup action may run.

        Startup actions run before combat is released, so this deliberately
        does not depend on ``should_run_combat()``. It does require a clear
        area and a SAFE danger state; otherwise buffs/timers remain pending.
        """
        if (
            not self.startup_area_clear.is_set()
            or self.is_stopped()
            or self.pause_event.is_set()
            or self.sitting_event.is_set()
            or self.storage_event.is_set()
            or self.healing_event.is_set()
            or self.discovery_suspend.is_set()
            or self.gates.danger_sit_requested.is_set()
            or self.gates.critical_danger_requested.is_set()
        ):
            return False
        danger = self.danger_detector
        if danger is not None and not danger.is_safe_for_heal():
            return False
        return self.tracks.get_area_clear_candidate().clear

    def should_allow_danger_teleport(self) -> bool:
        return self.gates.should_allow_danger_teleport()

    def request_danger_sit(self) -> bool:
        """Request the sit worker to handle damage danger when a sit key exists.

        A missing sit key must not leave ``danger_sit_requested`` set forever,
        because that would permanently block combat without a worker able to
        consume the request.
        """
        try:
            sit_scan = int(getattr(self.config, "sit_on_low_sp_scan_code", 0))
        except (TypeError, ValueError):
            sit_scan = 0
        if sit_scan <= 0:
            return False
        self.gates.request_danger_sit()
        return True

    def pop_danger_sit_request(self) -> bool:
        return self.gates.pop_danger_sit_request()

    def request_critical_danger(self) -> None:
        self.gates.request_critical_danger()

    def pop_critical_danger(self) -> bool:
        return self.gates.pop_critical_danger()

    def should_run_discovery(self) -> bool:
        return self.gates.should_run_discovery()

    def should_run_tracking(self) -> bool:
        return self.gates.should_run_tracking()

    def is_stopped(self) -> bool:
        return self.gates.is_stopped()

    def mark_running(self) -> None:
        self.gates.mark_running()

    def begin_hunt_startup(
        self,
        *,
        require_buffs: bool = True,
        require_timers: bool = True,
    ) -> None:
        self.gates.startup.begin(
            require_buffs=require_buffs,
            require_timers=require_timers,
        )

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
