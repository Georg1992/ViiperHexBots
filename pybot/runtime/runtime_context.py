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
storage yes       yes         no     yes
heal    yes       yes         no     yes
======= ========= =========== ====== ======

Sit, storage, and heal are mutually exclusive. Discovery and tracking are
suspended while sitting, as well as during explicit stop/user pause or
teleport settle. Storage and healing keep observation active.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field


from pybot.runtime.capture.hunt_capture import HuntWindowCapture
from pybot.config.runtime import HuntRuntimeConfig
from pybot.runtime.control import RuntimeControl
from pybot.runtime.danger_detector import DangerController, DangerDetector
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
    # Assembled by the context factory so the gates and the detector always
    # observe the same session state.
    danger_detector: DangerDetector | None = field(default=None, repr=False)
    danger_controller: DangerController | None = field(default=None, repr=False)
    # Set by the composition root. Observation may cancel an in-flight input
    # operation, but this callback never performs gameplay input itself.
    cancel_gameplay_input: Callable[[], None] | None = field(
        default=None, repr=False,
    )
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
    def danger_wake(self) -> threading.Event:
        """Wake GameplayLoop so it can re-read danger facts immediately."""
        return self.gates.danger_wake

    @danger_wake.setter
    def danger_wake(self, event: threading.Event) -> None:
        self.gates.danger_wake = event

    @property
    def discovery_wake(self) -> threading.Event:
        return self.gates.discovery_wake

    @discovery_wake.setter
    def discovery_wake(self, event: threading.Event) -> None:
        self.gates.discovery_wake = event

    @property
    def tracking_wake(self) -> threading.Event:
        """Wake tracking when discovery publishes a candidate."""
        return self.gates.tracking_wake

    @tracking_wake.setter
    def tracking_wake(self, event: threading.Event) -> None:
        self.gates.tracking_wake = event

    @property
    def attack_wake(self) -> threading.Event:
        """Wake gameplay when tracking publishes a newly live target."""
        return self.gates.attack_wake

    @attack_wake.setter
    def attack_wake(self, event: threading.Event) -> None:
        self.gates.attack_wake = event

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
    def character_action_gate(self) -> object:
        """Shared stagger + buff priority between buff casts and timer presses."""
        return self.gates.character_action_gate

    @property
    def danger_escape_active(self) -> threading.Event:
        """True while any urgent danger escape owns the transition."""
        return self.gates.danger_escape_active

    @property
    def area_transition_lock(self):
        """Serialize area reset, discovery commit, and no-target decisions."""
        return self.gates.area_transition_lock

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

    def try_heal_if_allowed(self, allowed, action) -> str:
        """Atomically admit one skill heal and return its precise result."""
        return self.gates.try_heal_if_allowed(allowed, action)

    def should_run_timers(self) -> bool:
        """True when skill timers may fire (lifecycle + no active danger).

        Timers keep running during storage and healing, but must never press
        a key while the character is taking damage. The lifecycle gate alone
        only knows about queued escape requests; the shared danger detector
        adds the recent-damage safety window used by buffs and heals.
        """
        if not self.gates.should_run_timers():
            return False
        danger = self.danger_detector
        if danger is None:
            return True
        return bool(danger.is_safe_for_heal())

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

    def should_run_custom_heal_actions(self) -> bool:
        """True when the custom skill heal may run.

        A skill heal is free, so it runs whenever the character is safe and
        combat is allowed (``should_run_character_actions``) — or during the
        short post-teleport grace window, the intended heal moment right after
        a danger/area teleport even when the last hit landed within the
        damage-quiet window. Without this, a damage cascade after a critical
        teleport (recent damage keeps ``is_safe_for_heal`` false) never lets
        the bot heal at critically low HP. Session gates (sit/storage/heal/
        pause/stop) and an in-flight teleport still block the cast.
        """
        try:
            heal_settings = getattr(self.config, "custom_behavior", None)
            heal_scan = int(getattr(heal_settings, "heal_scan_code", 0) or 0)
            heal_button = str(getattr(heal_settings, "heal_button", "") or "").strip()
        except (AttributeError, TypeError, ValueError):
            heal_scan = 0
        if heal_scan <= 0 or not heal_button:
            return False
        if self.in_post_teleport_heal_window():
            return not (
                self.is_stopped()
                or self.pause_event.is_set()
                or self.sitting_event.is_set()
                or self.storage_event.is_set()
                or self.healing_event.is_set()
                or self.discovery_suspend.is_set()
            )
        return self.should_run_character_actions()

    def should_run_startup_actions(self) -> bool:
        """True when a new-hunt startup action may run.

        Startup actions run before combat is released, so this deliberately
        does not depend on ``should_run_combat()``. A fresh hunt is trusted to
        start at a safe location (``area_clear`` is pre-satisfied by
        ``begin``); recovered hunts wait for a scan to confirm the landing
        area. Actions also require a SAFE danger state; otherwise buffs/timers
        remain pending.
        """
        if (
            not self.startup_area_clear.is_set()
            or self.is_stopped()
            or self.pause_event.is_set()
            or self.sitting_event.is_set()
            or self.storage_event.is_set()
            or self.healing_event.is_set()
            or self.discovery_suspend.is_set()
        ):
            return False
        danger = self.danger_detector
        if danger is not None and not danger.is_safe_for_heal():
            return False
        return self.tracks.get_area_clear_candidate().clear

    def should_allow_danger_teleport(self) -> bool:
        return self.gates.should_allow_danger_teleport()

    def begin_danger_transition(self, *, allow_sitting: bool = False) -> bool:
        return self.gates.begin_danger_transition(allow_sitting=allow_sitting)

    def end_danger_transition(self) -> None:
        self.gates.end_danger_transition()

    def finish_danger_transition(self, *, seated: bool) -> None:
        self.gates.finish_danger_transition(seated=seated)

    def should_run_discovery(self) -> bool:
        return self.gates.should_run_discovery()

    def should_run_tracking(self) -> bool:
        """Return whether local tracking may continue independently of combat."""
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

    def clear_post_teleport_heal(self) -> None:
        self.gates.clear_post_teleport_heal()

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
        # Keep direct callers safe too; TeleportController additionally holds
        # this lock while resetting the strategy marker so the complete area
        # transition is one coherent lifecycle boundary.
        with self.gates.area_transition_lock:
            self.tracks.area_reset()
            # A track-created wake belongs to the old area. Clear it together
            # with the track store so the next hunt cannot spin on a stale
            # producer signal before discovery confirms the new screen.
            self.attack_wake.clear()
            self.tracking_wake.clear()
            # Local visual anchors and LK points are screen-local. Clear them
            # at the same epoch boundary as the Track store.
            clear_states = getattr(self.tracker, "clear_track_states", None)
            if callable(clear_states) and isinstance(self.tracker, DetectorSession):
                clear_states()
            self.policy.reset()
            self.validation.log_area_reset(reason)
