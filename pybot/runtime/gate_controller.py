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
        # Set while heal-until-full after critical danger TP — combat idles; timers keep going.
        self.healing_event = threading.Event()
        self._sit_storage_lock = threading.Lock()

    # ── Gate queries ─────────────────────────────────────────────

    def is_stopped(self) -> bool:
        return self.stop_event.is_set()

    def should_run_workers(self) -> bool:
        """True when discovery, tracking, and skill timers may run.

        False while stopped, user-paused, or sitting. Storage/healing do not
        clear this (timers and HP restore keep running).
        """
        return (
            not self.stop_event.is_set()
            and not self.pause_event.is_set()
            and not self.sitting_event.is_set()
        )

    def should_run_combat(self) -> bool:
        """True when attack may run (workers running and not in storage/heal)."""
        return (
            self.should_run_workers()
            and not self.storage_event.is_set()
            and not self.healing_event.is_set()
        )

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

    # ── Pause / resume ───────────────────────────────────────────

    def mark_running(self) -> None:
        """Workers may run; wake any thread blocked in ``wait_while_stopped_or_paused``."""
        self.pause_event.clear()
        if not self.sitting_event.is_set():
            self.resume_gate.set()

    def mark_paused(self) -> None:
        """Workers must idle until ``mark_running``."""
        self.pause_event.set()
        self.resume_gate.clear()

    # ── Sit lifecycle ────────────────────────────────────────────

    def try_begin_sit_ops(self) -> bool:
        """Acquire sit pause (hunt + timers). False if sit/storage/heal held."""
        with self._sit_storage_lock:
            if (
                self.sitting_event.is_set()
                or self.storage_event.is_set()
                or self.healing_event.is_set()
            ):
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
            if not self.pause_event.is_set() and not self.stop_event.is_set():
                self.resume_gate.set()

    # ── Storage lifecycle ────────────────────────────────────────

    def try_begin_storage_ops(self) -> bool:
        """Acquire storage session (combat only). False if sit/storage/heal held."""
        with self._sit_storage_lock:
            if (
                self.sitting_event.is_set()
                or self.storage_event.is_set()
                or self.healing_event.is_set()
            ):
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
        with self._sit_storage_lock:
            self.storage_event.clear()
        # Wake combat (wait_while_combat_blocked polls resume_gate),
        # and discovery/tracking (blocked by storage/heal in should_run_*).
        self.resume_gate.set()
        self.discovery_wake.set()

    # ── Heal lifecycle ───────────────────────────────────────────

    def try_begin_heal_ops(self) -> bool:
        """Acquire heal-until-full session (combat only). False if sit/storage/heal held."""
        with self._sit_storage_lock:
            if (
                self.sitting_event.is_set()
                or self.storage_event.is_set()
                or self.healing_event.is_set()
            ):
                return False
            self.healing_event.set()
            return True

    def begin_heal_ops(self) -> bool:
        """Wait until heal ops can start. False if stopped first."""
        while not self.stop_event.is_set():
            if self.try_begin_heal_ops():
                return True
            self.stop_event.wait(WORKER_POLL_INTERVAL_S)
        return False

    def end_heal_ops(self) -> None:
        """Release heal session; combat may resume."""
        with self._sit_storage_lock:
            self.healing_event.clear()
        self.resume_gate.set()
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
            # Storage/heal do not clear resume_gate; poll stop/sit wake.
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
