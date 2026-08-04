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

from pybot.runtime.constants import (
    SKILL_TIMER_STAGGER_MS,
    WORKER_POLL_INTERVAL_S,
)
from pybot.runtime.startup_sequence import HuntStartupSequence


class CharacterActionGate:
    """Shared 500ms stagger + buff priority across buff casts and timer presses.

    SelfBuffWorker and SkillTimerWorker run on independent threads with no
    shared scheduling. When a buff and a timer come due in the same instant
    (for example three 240s configs syncing up), this gate makes them
    cooperate: a single stagger window spaces every character keypress, and
    a ``buff burst`` signal makes timers yield while buffs are casting so
    buffs fire first.

    Priority is best-effort: the buff worker raises its burst flag before
    claiming the slot, and timer claims are refused while it is up, so a
    simultaneous due-time race is overwhelmingly won by the buff. There is
    still a microsecond window where a timer claim beats the flag raise;
    both workers then simply space out by the shared stagger window.
    """

    def __init__(self, stagger_ms: int = SKILL_TIMER_STAGGER_MS) -> None:
        self._lock = threading.Lock()
        self._stagger_ms = max(0, int(stagger_ms))
        # ``None`` means no prior action yet. Zero is a valid monotonic test
        # timestamp, so it must not be used as the sentinel.
        self._last_action_ms: int | None = None
        self._buff_burst_active = False

    def note_action(self, now_ms: int) -> None:
        """Record a completed buff/timer keypress time."""
        with self._lock:
            if self._last_action_ms is None or now_ms > self._last_action_ms:
                self._last_action_ms = now_ms

    def try_claim(self, *, is_buff: bool, now_ms: int) -> bool:
        """Atomically claim the next character-keypress slot if it is open.

        A slot is open when the shared stagger window has elapsed since the
        last buff/timer keypress and (for timers) no buff burst is pending.
        A successful claim records ``now_ms`` as the new shared last-action
        time so the other worker waits out the window.
        """
        with self._lock:
            if not is_buff and self._buff_burst_active:
                return False
            if (
                self._last_action_ms is not None
                and now_ms - self._last_action_ms < self._stagger_ms
            ):
                return False
            self._last_action_ms = now_ms
            return True

    def stagger_remaining_ms(self, now_ms: int) -> int:
        """Milliseconds until the shared stagger window reopens (0 if open)."""
        with self._lock:
            if self._last_action_ms is None:
                return 0
            return max(0, self._stagger_ms - (now_ms - self._last_action_ms))

    def begin_buff_burst(self) -> None:
        """Mark a buff burst (due buffs about to cast) so timers yield."""
        with self._lock:
            self._buff_burst_active = True

    def end_buff_burst(self) -> None:
        """Clear the buff burst after all due buffs cast or the burst aborts."""
        with self._lock:
            self._buff_burst_active = False

    def buff_burst_active(self) -> bool:
        """Return whether a buff worker currently has priority over timers."""
        with self._lock:
            return self._buff_burst_active


class GateController:
    """Event gate logic: which workers may run, sit/storage/heal lifecycle, waits.

    Owns: stop_event, pause_event, resume_gate, sitting_event, storage_event,
    healing_event, discovery_wake, discovery_suspend, _sit_storage_lock.
    """

    def __init__(self, startup: HuntStartupSequence | None = None) -> None:
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
        # Set while heal-until-full — combat idles; discovery/tracking/timers keep running.
        self.healing_event = threading.Event()
        self._sit_storage_lock = threading.Lock()
        # Serializes area transitions with discovery's commit boundary and
        # no-target decisions. Capture/detection may run outside this lock;
        # state publication and reset are one deterministic transaction.
        self.area_transition_lock = threading.RLock()
        # Shared cooldown for every HP-healing input path. Custom mob healing
        # and the HP-item worker must not both decide to heal in one window.
        self._last_heal_action_mono = 0.0
        # Monotonic deadline: after teleport settle, heal freely until this time.
        self._post_teleport_heal_until = 0.0
        # Shared stagger + buff priority between buff casts and skill-timer
        # presses (both are character keypresses on separate threads).
        self.character_action_gate = CharacterActionGate()
        # Startup milestones and hunt generations belong to the dedicated
        # sequence object, not to this general lifecycle gate.
        self.startup = HuntStartupSequence() if startup is None else startup
        # Set by DangerDetector for any HP drop; the sit worker owns seated
        # damage and filters ordinary hunting hits.
        self.danger_sit_requested = threading.Event()
        # Independent critical-danger signal so hunting can escape even when
        # low-SP sitting is disabled or no sit key is configured.
        self.critical_danger_requested = threading.Event()
        # True from any danger-escape ownership claim through teleport settle.
        # This is intentionally separate from sitting_event: the escape holds
        # the input gate but is not necessarily SP recovery.
        self.danger_escape_active = threading.Event()
        # Unlike danger_escape_active, this marker is exclusively owned by the
        # critical hunting escape. Active storage must abort only for this
        # emergency, not for the seated recovery escape's generic marker.
        self.critical_danger_escape_active = threading.Event()
        # Sessions (sit/storage/heal) that a critical escape overrode. Used
        # only to bound the pre-teleport wait for the preempted owners to
        # release; teardown always releases the gate the escape itself set.
        self._preempted_sessions = (False, False, False)
        # A seated toggle could not be undone during worker cleanup. Runtime
        # shutdown must retry this before releasing input ownership.
        self.sit_cleanup_unresolved = threading.Event()

    # ── Gate queries ─────────────────────────────────────────────

    def is_stopped(self) -> bool:
        return self.stop_event.is_set()

    def should_run_workers(self) -> bool:
        """True when discovery, tracking, and skill timers may run.

        False while stopped, user-paused, or sitting. Storage/healing do not
        clear this; their combat-specific gates are checked separately.
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
            and not self.danger_sit_requested.is_set()
            and not self.critical_danger_requested.is_set()
            and not self.danger_escape_active.is_set()
            and self.startup.is_combat_ready()
        )

    def should_run_timers(self) -> bool:
        """True when skill timers may fire.

        Timers keep running during storage and healing. Sitting, user pause,
        danger handling, an in-flight teleport, and an active escape suspend
        them.
        """
        return (
            self.should_run_workers()
            and not self.discovery_suspend.is_set()
            and not self.danger_sit_requested.is_set()
            and not self.critical_danger_requested.is_set()
            and not self.danger_escape_active.is_set()
        )

    def should_allow_danger_teleport(self) -> bool:
        """True when a danger teleport may run.

        Critical danger overrides every session (sit/storage/heal): only an
        explicit stop or user pause may hold it back. Session preemption is
        managed by the escape claim itself.
        """
        return not self.stop_event.is_set() and not self.pause_event.is_set()

    def should_run_discovery(self) -> bool:
        """True when discovery may scan (workers running and not storage).

        Heal does not block discovery — tracks stay fresh while topping up HP.
        """
        return (
            self.should_run_workers()
            and not self.storage_event.is_set()
            and not self.danger_sit_requested.is_set()
        )

    def should_run_tracking(self) -> bool:
        """True when tracking may tick outside an owned session transition.

        Tracking may continue during healing, but it must stop as soon as a
        sit/danger session owns the character or a teleport transition is in
        flight. Otherwise it can ingest a pre-teleport discovery candidate
        between the critical worker claiming the gate and the area reset.
        """
        return (
            self.should_run_workers()
            and not self.storage_event.is_set()
            and not self.discovery_suspend.is_set()
            and not self.danger_sit_requested.is_set()
            and not self.critical_danger_requested.is_set()
            and not self.danger_escape_active.is_set()
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

    def perform_input_if_allowed(self, allowed, action) -> bool:
        """Admit one short hunt input action against session transitions.

        The existing session lock is also the ownership boundary for sit,
        storage, and healing. Hold it only for the final gate check and the
        already-atomic input operation; callers perform capture and waits
        outside this method.
        """
        with self._sit_storage_lock:
            if not allowed():
                return False
            result = action()
            # Existing input workers treat only an explicit False as a
            # rejected action; lightweight backends commonly return None.
            return result is not False

    def perform_heal_if_allowed(self, allowed, action, *, cooldown_s: float = 1.0) -> bool:
        """Admit one healing input and share its cooldown across workers."""
        with self._sit_storage_lock:
            if not allowed():
                return False
            now = time.monotonic()
            if now - self._last_heal_action_mono < max(0.0, cooldown_s):
                return False
            result = action()
            if result is not False:
                self._last_heal_action_mono = time.monotonic()
                return True
            return False

    def request_danger_sit(self) -> None:
        """Ask the sit worker to move safe, sit, recover, and restart hunt."""
        with self._sit_storage_lock:
            self.danger_sit_requested.set()
            self.resume_gate.set()

    def pop_danger_sit_request(self) -> bool:
        """Consume one pending danger-driven sit request atomically."""
        with self._sit_storage_lock:
            if not self.danger_sit_requested.is_set():
                return False
            self.danger_sit_requested.clear()
            return True

    def request_critical_danger(self) -> None:
        """Queue a critical hunting escape independent of sit configuration."""
        with self._sit_storage_lock:
            self.critical_danger_requested.set()
            self.resume_gate.set()

    def pop_critical_danger(self) -> bool:
        """Consume one pending critical hunting escape atomically."""
        with self._sit_storage_lock:
            if not self.critical_danger_requested.is_set():
                return False
            self.critical_danger_requested.clear()
            return True

    def begin_danger_escape(self) -> bool:
        """Claim an urgent escape from an already-owned session."""
        with self._sit_storage_lock:
            if self.danger_escape_active.is_set():
                return False
            self.danger_escape_active.set()
            return True

    def try_begin_critical_escape_ops(self, *, override: bool = False) -> bool:
        """Atomically claim the critical escape and its input gate.

        The critical worker is not an SP-recovery session. Claim both markers
        under one lock so HP polling cannot see ``sitting_event`` alone and
        enqueue a sit request in the pre-teleport window.

        With ``override=True`` the escape preempts any active session (sit
        recovery, storage UI, heal). Critical danger has the highest priority:
        the preempted owners see ``danger_escape_active`` on their next loop
        and abandon without touching input or gates, and the critical worker
        waits (bounded) for storage/heal to release before pressing the
        teleport key.
        """
        with self._sit_storage_lock:
            if self.danger_escape_active.is_set():
                return False
            if not override and self._session_held():
                return False
            self._preempted_sessions = (
                self.sitting_event.is_set(),
                self.storage_event.is_set(),
                self.healing_event.is_set(),
            )
            self.danger_escape_active.set()
            self.critical_danger_escape_active.set()
            self.sitting_event.set()
            self.resume_gate.clear()
            return True

    def preempted_sessions(self) -> tuple[bool, bool, bool]:
        """Sessions (sit, storage, heal) the current escape overrode."""
        return self._preempted_sessions

    def wait_for_preempted_session_release(self, timeout_s: float) -> bool:
        """Block until sessions preempted by the escape release their gates.

        The escape itself holds ``sitting_event`` (set by its own claim), so
        only preempted storage/heal owners are waited on: storage must close
        its UI panels before the teleport key is pressed, otherwise the wing
        is wasted against an open panel. A preempted sit session needs no
        release — the character can teleport while seated and its worker
        stops touching input once it sees the escape.

        The preempted snapshot is overwritten by the next claim, so it only
        ever describes the escape currently in flight.

        Returns True when nothing remains to release. Exits early on stop or
        user pause (the caller re-checks both after this wait).
        """
        preempted = self._preempted_sessions
        if not (preempted[1] or preempted[2]):
            return True
        deadline = time.monotonic() + max(0.0, timeout_s)
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                return False
            with self._sit_storage_lock:
                still_held = (
                    (preempted[1] and self.storage_event.is_set())
                    or (preempted[2] and self.healing_event.is_set())
                )
                if not still_held:
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self.stop_event.wait(min(WORKER_POLL_INTERVAL_S, remaining))
        return False

    def end_danger_escape(self) -> None:
        """Release the explicit danger-escape phase after teleport handling."""
        with self._sit_storage_lock:
            self.danger_escape_active.clear()

    def end_critical_escape_ops(self) -> None:
        """Release a critical escape without starting a sit/hunt generation.

        Critical danger temporarily borrows the input exclusion gate, but it is
        not SP recovery. Calling ``end_sit_ops`` here would re-arm startup
        milestones and make the attack loop appear stale after a successful
        danger teleport.
        """
        with self._sit_storage_lock:
            self.danger_escape_active.clear()
            self.critical_danger_escape_active.clear()
            self.sitting_event.clear()
            self._restore_resume_gate()

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

    def end_sit_ops(self, *, trusted_clear: bool = True) -> None:
        """Release sit pause and begin a fresh hunt cycle.

        Standing after SP recovery is a new hunt start: normal timers fire
        again, then character buffs replay in order. The recovery session only
        completes at a spot the character sat through without damage (a
        bot-chosen quiet area or a safe-key escape landing), so by default the
        area is trusted clear: startup buffs/timers run immediately instead of
        waiting for the first discovery scan, and combat resumes without a
        dead window. Pass ``trusted_clear=False`` for landings that were not
        verified safe (e.g. a random fly-wing escape). The generation/event
        pair lets those independent workers coordinate without relying on
        thread start order. Discovery is explicitly woken after the new
        generation is published so the resumed hunt does not wait for the
        full cadence.
        """
        with self._sit_storage_lock:
            # Reset the generation and startup milestones first. The sit gate
            # remains held until this completes, so workers cannot observe a
            # new hunt with the previous hunt's completion events.
            self.startup.begin_new_hunt(trusted_clear=trusted_clear)
            self.sitting_event.clear()
            self._restore_resume_gate()
        # The discovery worker may be asleep in its cadence wait while the sit
        # session owns the worker gate. Wake it only after the gate and startup
        # generation are coherent, making the first post-recovery scan prompt.
        self.discovery_wake.set()

    def mark_sit_cleanup_unresolved(self) -> None:
        """Keep runtime ownership until a seated state is explicitly undone."""
        self.sit_cleanup_unresolved.set()

    def clear_sit_cleanup_unresolved(self) -> None:
        """Record that shutdown successfully undid the seated toggle."""
        self.sit_cleanup_unresolved.clear()

    # ── Storage lifecycle ────────────────────────────────────────

    def try_begin_storage_ops(self) -> bool:
        """Acquire storage only when no higher-priority action is pending."""
        with self._sit_storage_lock:
            # This check must live inside the same lock as session acquisition.
            # A flag-only pre-check in ItemsToStorageWorker leaves a race where
            # storage wins just before danger/sit claims the character.
            if (
                self._session_held()
                or self.critical_danger_requested.is_set()
                or self.danger_sit_requested.is_set()
                or self.danger_escape_active.is_set()
            ):
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
        """Release heal session; combat/timers may resume."""
        with self._sit_storage_lock:
            self.healing_event.clear()
            self._restore_resume_gate()
        self.discovery_wake.set()

    # ── Post-teleport heal window ────────────────────────────────

    def mark_post_teleport_heal(self, duration_s: float) -> None:
        """Open the post-teleport heal window (mobs ignore the character briefly)."""
        self._post_teleport_heal_until = time.monotonic() + duration_s

    def in_post_teleport_heal_window(self) -> bool:
        """True for a short time after teleport settle completes."""
        return time.monotonic() < self._post_teleport_heal_until

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

    def wait_while_user_paused(self, timeout_s: float) -> bool:
        """Block while the user-pause flag is set (ignores sit/heal/storage).

        Returns True when not paused and not stopped. Used inside sit sessions
        where ``should_run_workers`` is false because ``sitting_event`` is held.
        """
        deadline = time.monotonic() + timeout_s
        while not self.stop_event.is_set():
            if not self.pause_event.is_set():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return not self.pause_event.is_set()
            self.stop_event.wait(min(WORKER_POLL_INTERVAL_S, remaining))
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
