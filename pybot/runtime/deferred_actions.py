"""Small serialized scheduler for gameplay actions that may be deferred.

The scheduler deliberately does not own a thread.  Monotonic deadlines are
observed when the gameplay owner reaches a scheduling point; an action that
becomes due while a higher-priority session owns the character is latched as
pending and is not silently re-armed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from pybot.runtime.hunt_tracks import monotonic_ms


@dataclass(frozen=True)
class ActionExecution:
    """Result returned by a scheduled action callback.

    ``completed_keys`` supports one serialized callback completing a small
    group, such as a startup buff sequence.  The per-key timestamps preserve
    the important invariant that each deadline restarts from actual success,
    not from expiry or from the end of an unrelated action.
    """

    success: bool
    completed_keys: tuple[str, ...] = ()
    completed_at_ms: dict[str, int] = field(default_factory=dict)


@dataclass
class DeferredAction:
    """One periodic action and its explicit scheduling state."""

    key: str
    interval_ms: int
    priority: int
    execute: Callable[[], ActionExecution | bool]
    order: int
    ready: Callable[[], bool] | None = None
    due_when: Callable[[], bool] | None = None
    due_on_generation: bool = True
    next_due_ms: int = 0
    expired: bool = False
    pending: bool = False
    last_executed_ms: int | None = None

    def is_due(self, now_ms: int) -> bool:
        """Whether the deadline has elapsed, without changing pending state."""
        return int(now_ms) >= self.next_due_ms



class DeferredActionScheduler:
    """Own due/pending state for the single gameplay action owner."""

    def __init__(self, *, clock: Callable[[], int] = monotonic_ms) -> None:
        self._actions: dict[str, DeferredAction] = {}
        self._clock = clock
        self._order = 0
        self._generation: int | None = None
        self._retry_required = False

    def register(
        self,
        key: str,
        *,
        interval_ms: int,
        priority: int,
        execute: Callable[[], ActionExecution | bool],
        ready: Callable[[], bool] | None = None,
        due_when: Callable[[], bool] | None = None,
        due_on_generation: bool = True,
    ) -> None:
        """Register one action, initially due immediately."""
        if key in self._actions:
            raise ValueError(f"duplicate deferred action key: {key}")
        self._actions[key] = DeferredAction(
            key=key,
            interval_ms=max(1, int(interval_ms)),
            priority=int(priority),
            execute=execute,
            order=self._order,
            ready=ready,
            due_when=due_when,
            due_on_generation=due_on_generation,
        )
        self._order += 1

    def sync_generation(self, generation: int, *, now_ms: int) -> None:
        """Make all generation-sensitive actions due for a fresh hunt cycle."""
        generation = int(generation)
        if self._generation == generation:
            return
        self._generation = generation
        for action in self._actions.values():
            action.last_executed_ms = None
            action.expired = bool(action.due_on_generation)
            if action.due_on_generation:
                action.next_due_ms = int(now_ms)
                action.pending = True
            else:
                action.next_due_ms = int(now_ms) + action.interval_ms
                action.pending = False

    def observe(self, *, now_ms: int) -> None:
        """Latch every elapsed deadline without moving it forward."""
        now_ms = int(now_ms)
        for action in self._actions.values():
            if action.due_when is not None and not action.due_when():
                # Condition-driven actions (HP restoration) are no longer
                # pending when the condition has cleared; no timer expiry can
                # manufacture work while the character is healthy.
                action.expired = False
                action.pending = False
                continue
            if now_ms >= action.next_due_ms and (
                action.due_on_generation or action.last_executed_ms is not None
            ):
                action.expired = True
                action.pending = True

    def pending_actions(self, *, now_ms: int) -> tuple[DeferredAction, ...]:
        """Return pending actions in stable priority/configuration order."""
        self.observe(now_ms=now_ms)
        return tuple(
            sorted(
                (action for action in self._actions.values() if action.pending),
                key=lambda action: (action.priority, action.order),
            )
        )

    def run_pending(self, *, now_ms: int, max_actions: int | None = None) -> bool:
        """Try pending actions in order; stop at the first failed/deferred one.

        Stopping at the first failure keeps priority meaningful and avoids a
        lower-priority action bypassing a higher-priority action that is still
        unsafe or retryable.
        """
        ran = False
        self._retry_required = False
        for index, action in enumerate(self.pending_actions(now_ms=now_ms)):
            if max_actions is not None and index >= max_actions:
                break
            # A pending action can remain unsafe while an independent lower
            # priority action is safe. Keep the pending bit, but do not let a
            # blocked maintenance action starve unrelated work.
            if not action.pending:
                continue
            if action.ready is not None and not action.ready():
                self._retry_required = True
                continue
            try:
                result = action.execute()
            except Exception:
                # The action remains expired/pending. The gameplay boundary
                # logs the exception and retries it on a later safe tick.
                self._retry_required = True
                raise
            if isinstance(result, ActionExecution):
                success = result.success
                completed = result.completed_keys
                completed_at = result.completed_at_ms
            else:
                success = bool(result)
                completed = (action.key,) if success else ()
                completed_at = {}
            if not success:
                self._retry_required = True
                break
            if not completed:
                completed = (action.key,)
            executed_at = self._clock()
            for key in completed:
                self.mark_executed(
                    key,
                    at_ms=completed_at.get(key, executed_at),
                )
            ran = True
        return ran

    def requires_retry(
        self,
        *,
        max_priority: int,
        ignore_keys: set[str] | frozenset[str] = frozenset(),
    ) -> bool:
        """Whether a due/unsafe action needs retry before lower-priority work.

        ``ignore_keys`` is for condition-driven maintenance that is useful when
        admissible but must never freeze unrelated gameplay when temporarily
        unsafe (for example an HP item outside its post-teleport window).
        """
        if not getattr(self, "_retry_required", False):
            return False
        return any(
            action.pending
            and action.priority <= max_priority
            and action.key not in ignore_keys
            for action in self._actions.values()
        )

    def mark_pending(self, key: str) -> None:
        """Latch an action explicitly when its current observation says it is due."""
        action = self._actions[key]
        action.expired = True
        action.pending = True

    def mark_executed(self, key: str, *, at_ms: int) -> None:
        """Restart one action only after its callback reports success."""
        action = self._actions[key]
        executed_at = int(at_ms)
        action.expired = False
        action.pending = False
        action.last_executed_ms = executed_at
        action.next_due_ms = executed_at + action.interval_ms

    def seed_executed(self, key: str, *, at_ms: int) -> None:
        """Seed a deadline from a startup action already executed successfully."""
        action = self._actions.get(key)
        if action is not None:
            self.mark_executed(key, at_ms=at_ms)

    def get(self, key: str) -> DeferredAction:
        return self._actions[key]

    def statuses(self, *, now_ms: int) -> tuple[DeferredAction, ...]:
        """Snapshot all states for diagnostics/tests without changing state."""
        return tuple(self._actions.values())

    def __iter__(self) -> Iterable[DeferredAction]:
        return iter(self._actions.values())
