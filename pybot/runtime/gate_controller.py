"""GateController — event gates and worker lifecycle orchestration.

Owns the event gates that govern which workers may run (stop, pause, sit,
storage, healing) and provides the lifecycle operations (begin/end) and wait
helpers that workers use to coordinate.

HuntRuntimeContext holds one instance and delegates its gate methods here.
This keeps the dataclass from accumulating business logic.
"""

from __future__ import annotations

import threading
import time

from pybot.runtime.constants import WORKER_POLL_INTERVAL_S


class GateController:
    """Event gate logic: which workers may run, sit/storage/heal lifecycle, waits.

    Owns: stop_event, pause_event, resume_gate, sitting_event, storage_event,
    healing_event, discovery_wake, discovery_suspend, _sit_storage_lock.
    """

    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.resume_gate = threading.Event()
        self.discovery_wake = threading.Event()
        # Set for the whole claim → teleport key → settle delay window so the
        # 1s discovery cadence cannot scan mid-teleport and falsely confirm clear.
        self.discovery_suspend = threading.Event()
        # Set while regenerating SP (sit) — hunt + skill timers idle.
        self.sitting_event = threading.Event()
        # Set while ItemsToStorage / GetFlyWings runs — combat idles; timers keep going.
        self.storage_event = threading.Event()
        # Set while heal-until-full — combat/discovery/tracking/timers idle; HP worker runs.
        self.healing_event = threading.Event()
        self._sit_storage_lock = threading.Lock()

    # ── Gate queries ─────────────────────────────────────────────

    def is_stopped(self) -> bool:
        return self.stop_event.is_set()

    def should_run_workers(self) -> bool:
        """True when discovery, tracking, and skill timers may run.

        False while stopped, user-paused, or sitting. Storage/healing do not
        clear this (HP restore keeps running; timers have their own gate).
        """
        return (
            not self.stop_event.is_set()
            and not self.pause_event.is_set()
            and not self.sitting_event.is_set()
        )

    def should_run_combat(self) -> bool:
        """True when attack may run (workers running and not in storage/heal/TP)."""
        return (
            self.should_run_workers()
            and not self.storage_event.is_set()
            and not self.healing_event.is_set()
            and not self.discovery_suspend.is_set()
        )

    def should_run_timers(self) -> bool:
        """True when skill timers may fire.

        Timers keep running during storage, but pause during heal so they
        cannot steal the skill bar mid-heal cast.
        """
        return self.should_run_workers() and not self.healing_event.is_set()

    def should_allow_danger_teleport(self) -> bool:
        """True when danger TP may run.

        Heal does not block this — danger teleport has priority over healing.
        Sit/storage/pause/stop still block.
        """
        return self.should_run_workers() and not self.storage_event.is_set()

    def should_run_discovery(self) -> bool:
        """True when discovery may scan (workers running and not storage/heal)."""
        return (
            self.should_run_workers()
            and not self.storage_event.is_set()
            and not self.healing_event.is_set()
        )

    def should_run_tracking(self) -> bool:
        """True when tracking may tick (workers running and not storage/heal)."""
        return (
            self.should_run_workers()
            and not self.storage_event.is_set()
            and not self.healing_event.is_set()
        )

    def _session_held(self) -> bool:
        return (
            self.sitting_event.is_set()
            or self.storage_event.is_set()
            or self.healing_event.is_set()
        )

    def _restore_resume_gate(self) -> None:
        """Set resume_gate when nothing is holding combat/workers paused."""
        if (
            not self.pause_event.is_set()
            and not self.stop_event.is_set()
            and not self._session_held()
        ):
            self.resume_gate.set()

    # ── Pause / resume ───────────────────────────────────────────

    def mark_running(self) -> None:
        """Workers may run; wake any thread blocked in ``wait_while_stopped_or_paused``."""
        self.pause_event.clear()
        self._restore_resume_gate()

    def mark_paused(self) -> None:
        """Workers must idle until ``mark_running``."""
        self.pause_event.set()
        self.resume_gate.clear()

    # ── Sit lifecycle ────────────────────────────────────────────

    def try_begin_sit_ops(self) -> bool:
        """Acquire sit pause (hunt + timers). False if sit/storage/heal held."""
        with self._sit_storage_lock:
            if self._session_held():
                return False
            self.sitting_event.set()
            self.resume_gate.clear()
            return True

    def begin_sit_ops(self) -> bool:
        """Wait until sit ops can start. False if stopped first."""
        while not self.stop_event.is_set():
            if self.try_begin_sit_ops():
                return True
            self.stop_event.wait(WORKER_POLL_INTERVAL_S)
        return False

    def end_sit_ops(self) -> None:
        """Release sit pause; restore resume_gate unless user-paused/stopped."""
        with self._sit_storage_lock:
            self.sitting_event.clear()
            self._restore_resume_gate()

    # ── Storage lifecycle ────────────────────────────────────────

    def try_begin_storage_ops(self) -> bool:
        """Acquire storage session (combat only). False if sit/storage/heal held."""
        with self._sit_storage_lock:
            if self._session_held():
                return False
            self.storage_event.set()
            self.resume_gate.clear()
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
        with self._sit_storage_lock:
            self.storage_event.clear()
            self._restore_resume_gate()
        # Wake discovery/tracking blocked on storage_event / discovery_wake.
        self.discovery_wake.set()

    # ── Heal lifecycle ───────────────────────────────────────────

    def try_begin_heal_ops(self) -> bool:
        """Acquire heal-until-full session (combat only). False if sit/storage/heal held."""
        with self._sit_storage_lock:
            if self._session_held():
                return False
            self.healing_event.set()
            self.resume_gate.clear()
            return True

    def begin_heal_ops(self) -> bool:
        """Wait until heal ops can start. False if stopped or paused first."""
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                return False
            if self.try_begin_heal_ops():
                return True
            self.stop_event.wait(WORKER_POLL_INTERVAL_S)
        return False

    def end_heal_ops(self) -> None:
        """Release heal session; combat/discovery/tracking/timers may resume."""
        with self._sit_storage_lock:
            self.healing_event.clear()
            self._restore_resume_gate()
        self.discovery_wake.set()

    # ── Wait helpers ─────────────────────────────────────────────

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
        """Block while sit/pause/storage/heal holds combat. True if combat may run."""
        deadline = time.monotonic() + timeout_s
        while not self.stop_event.is_set():
            if self.should_run_combat():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self.should_run_combat()
            # begin_sit/storage/heal clear resume_gate; end_* sets it to wake us.
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

    def wait_unless_paused_or_suspended(self, timeout_s: float) -> bool:
        """Wait up to *timeout_s* unless stop, pause, or discovery_suspend.

        Returns True only when the full duration elapsed without interruption.
        Used by heal so danger teleport and user pause can preempt cast delays.
        """
        deadline = time.monotonic() + timeout_s
        while not self.stop_event.is_set():
            if self.pause_event.is_set() or self.discovery_suspend.is_set():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            if self.stop_event.wait(min(WORKER_POLL_INTERVAL_S, remaining)):
                return False
        return False
