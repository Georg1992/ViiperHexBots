"""Deferred scheduled-action semantics."""

from __future__ import annotations

import unittest

from pybot.runtime.deferred_actions import DeferredActionScheduler


class DeferredActionSchedulerTests(unittest.TestCase):
    def test_expiry_becomes_pending_without_resetting_deadline(self) -> None:
        now = {"ms": 100}
        scheduler = DeferredActionScheduler(clock=lambda: now["ms"])
        calls: list[int] = []
        scheduler.register(
            "buff",
            interval_ms=1_000,
            priority=30,
            execute=lambda: calls.append(now["ms"]) or True,
        )
        scheduler.sync_generation(1, now_ms=0)
        self.assertEqual(scheduler.get("buff").next_due_ms, 0)
        self.assertTrue(scheduler.get("buff").expired)
        self.assertTrue(scheduler.get("buff").pending)

        now["ms"] = 250
        scheduler.run_pending(now_ms=now["ms"])

        self.assertEqual(calls, [250])
        self.assertFalse(scheduler.get("buff").expired)
        self.assertFalse(scheduler.get("buff").pending)
        self.assertEqual(scheduler.get("buff").next_due_ms, 1_250)

    def test_unsafe_action_stays_pending_until_safe(self) -> None:
        safe = {"value": False}
        calls: list[str] = []
        scheduler = DeferredActionScheduler(clock=lambda: 500)
        scheduler.register(
            "timer",
            interval_ms=100,
            priority=40,
            ready=lambda: safe["value"],
            execute=lambda: calls.append("executed") or True,
        )
        scheduler.sync_generation(1, now_ms=0)
        scheduler.run_pending(now_ms=500)
        self.assertEqual(calls, [])
        self.assertTrue(scheduler.get("timer").expired)
        self.assertTrue(scheduler.get("timer").pending)
        self.assertEqual(scheduler.get("timer").next_due_ms, 0)

        safe["value"] = True
        scheduler.run_pending(now_ms=500)
        self.assertEqual(calls, ["executed"])
        self.assertFalse(scheduler.get("timer").pending)

    def test_buff_and_timer_tick_during_teleport_suspend_without_executing(self) -> None:
        """Deadlines advance on black screens; keypresses wait for landing."""
        import threading

        suspend = threading.Event()
        suspend.set()
        calls: list[str] = []
        scheduler = DeferredActionScheduler(clock=lambda: 500)
        scheduler.register(
            "buff",
            interval_ms=100,
            priority=30,
            ready=lambda: not suspend.is_set(),
            execute=lambda: calls.append("buff") or True,
        )
        scheduler.register(
            "timer",
            interval_ms=100,
            priority=40,
            ready=lambda: not suspend.is_set(),
            execute=lambda: calls.append("timer") or True,
        )
        scheduler.sync_generation(1, now_ms=0)

        scheduler.run_pending(now_ms=500)

        self.assertEqual(calls, [])
        self.assertTrue(scheduler.get("buff").expired)
        self.assertTrue(scheduler.get("buff").pending)
        self.assertTrue(scheduler.get("timer").expired)
        self.assertTrue(scheduler.get("timer").pending)
        self.assertEqual(scheduler.get("buff").next_due_ms, 0)
        self.assertEqual(scheduler.get("timer").next_due_ms, 0)

        suspend.clear()
        scheduler.run_pending(now_ms=500)

        self.assertEqual(calls, ["buff", "timer"])
        self.assertFalse(scheduler.get("buff").pending)
        self.assertFalse(scheduler.get("timer").pending)

    def test_failure_does_not_restart_timer_or_allow_success_state(self) -> None:
        attempts = {"count": 0}
        scheduler = DeferredActionScheduler(clock=lambda: 900)

        def execute() -> bool:
            attempts["count"] += 1
            return attempts["count"] > 1

        scheduler.register("buff", interval_ms=100, priority=30, execute=execute)
        scheduler.sync_generation(1, now_ms=0)
        scheduler.run_pending(now_ms=900)
        state = scheduler.get("buff")
        self.assertTrue(state.pending)
        self.assertIsNone(state.last_executed_ms)
        self.assertEqual(state.next_due_ms, 0)

        scheduler.run_pending(now_ms=901)
        state = scheduler.get("buff")
        self.assertFalse(state.pending)
        self.assertEqual(state.last_executed_ms, 900)
        self.assertEqual(state.next_due_ms, 1_000)

    def test_multiple_expirations_drain_in_priority_then_registration_order(self) -> None:
        calls: list[str] = []
        scheduler = DeferredActionScheduler(clock=lambda: 200)
        for key, priority in (("normal", 40), ("urgent", 20), ("defensive", 30)):
            scheduler.register(
                key,
                interval_ms=100,
                priority=priority,
                execute=lambda key=key: calls.append(key) or True,
            )
        scheduler.sync_generation(1, now_ms=0)
        scheduler.run_pending(now_ms=200)
        self.assertEqual(calls, ["urgent", "defensive", "normal"])
        self.assertTrue(all(not action.pending for action in scheduler))

    def test_condition_driven_action_is_not_due_when_condition_is_clear(self) -> None:
        condition = {"low": False}
        calls: list[str] = []
        scheduler = DeferredActionScheduler(clock=lambda: 1_000)
        scheduler.register(
            "hp",
            interval_ms=100,
            priority=20,
            due_when=lambda: condition["low"],
            ready=lambda: True,
            execute=lambda: calls.append("heal") or True,
            due_on_generation=False,
        )
        scheduler.sync_generation(1, now_ms=0)
        scheduler.run_pending(now_ms=1_000)
        self.assertEqual(calls, [])
        self.assertFalse(scheduler.get("hp").expired)
        self.assertFalse(scheduler.get("hp").pending)

        condition["low"] = True
        scheduler.mark_pending("hp")
        scheduler.run_pending(now_ms=1_000)
        self.assertEqual(calls, ["heal"])

    def test_ignored_unsafe_action_does_not_require_retry(self) -> None:
        """Optional maintenance may remain pending without freezing gameplay."""
        scheduler = DeferredActionScheduler(clock=lambda: 500)
        scheduler.register(
            "hp_restore",
            interval_ms=100,
            priority=20,
            ready=lambda: False,
            execute=lambda: True,
            due_on_generation=False,
        )
        scheduler.sync_generation(1, now_ms=0)
        scheduler.mark_pending("hp_restore")
        scheduler.run_pending(now_ms=500)

        self.assertTrue(scheduler.get("hp_restore").pending)
        self.assertFalse(
            scheduler.requires_retry(max_priority=40, ignore_keys={"hp_restore"})
        )
        self.assertTrue(scheduler.requires_retry(max_priority=40))

    def test_successful_callback_uses_post_callback_timestamp(self) -> None:
        now = {"ms": 100}
        scheduler = DeferredActionScheduler(clock=lambda: now["ms"])

        def execute() -> bool:
            now["ms"] = 175
            return True

        scheduler.register("timer", interval_ms=50, priority=40, execute=execute)
        scheduler.sync_generation(1, now_ms=100)
        scheduler.run_pending(now_ms=100)
        self.assertEqual(scheduler.get("timer").last_executed_ms, 175)
        self.assertEqual(scheduler.get("timer").next_due_ms, 225)


if __name__ == "__main__":
    unittest.main()
