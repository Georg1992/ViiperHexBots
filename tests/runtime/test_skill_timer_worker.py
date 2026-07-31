"""Skill timer stagger and sit/pause re-arm behavior."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pybot.config.runtime import SkillTimerRuntime
from pybot.runtime.constants import SKILL_TIMER_STAGGER_MS
from pybot.runtime.workers.skill_timer_worker import SkillTimerWorker


class SkillTimerWorkerTests(unittest.TestCase):
    def test_staggers_due_timers_by_500ms(self) -> None:
        timers = (
            SkillTimerRuntime(button="f1", scan_code=59, interval_ms=60_000),
            SkillTimerRuntime(button="f2", scan_code=60, interval_ms=60_000),
        )
        stop = threading.Event()
        presses: list[int] = []
        clock = {"ms": 1_000_000}
        waits: list[float] = []

        def teleport_key(scan_code: int) -> None:
            presses.append(scan_code)
            if len(presses) >= 2:
                stop.set()

        def wait_paused(timeout_s: float) -> bool:
            waits.append(timeout_s)
            clock["ms"] += int(round(timeout_s * 1000))
            return not stop.is_set()

        ctx = SimpleNamespace(
            config=SimpleNamespace(skill_timers=timers),
            logger=SimpleNamespace(behavior=MagicMock()),
            stop_event=stop,
            resume_gate=threading.Event(),
            is_stopped=stop.is_set,
            should_run_workers=lambda: not stop.is_set(),
            should_run_timers=lambda: not stop.is_set(),
            wait_while_stopped_or_paused=wait_paused,
        )
        worker = SkillTimerWorker(ctx, SimpleNamespace(teleport_key=teleport_key))

        with patch(
            "pybot.runtime.workers.skill_timer_worker.monotonic_ms",
            side_effect=lambda: clock["ms"],
        ):
            worker.run()

        self.assertEqual(presses, [59, 60])
        self.assertEqual(len(waits), 1)
        self.assertAlmostEqual(waits[0], SKILL_TIMER_STAGGER_MS / 1000.0, places=3)

    def test_healing_does_not_disarm_or_rearm_timers(self) -> None:
        timers = (
            SkillTimerRuntime(button="f1", scan_code=59, interval_ms=10),
        )
        from pybot.runtime.gate_controller import GateController

        gates = GateController()
        gates.mark_running()
        presses: list[int] = []
        clock = {"ms": 0}

        def teleport_key(scan_code: int) -> None:
            presses.append(scan_code)
            if len(presses) == 1:
                self.assertTrue(gates.begin_heal_ops())
            else:
                gates.stop_event.set()

        def stop_wait(_timeout_s: float) -> bool:
            if not gates.stop_event.is_set():
                clock["ms"] += 1
            return gates.stop_event.is_set()

        ctx = SimpleNamespace(
            config=SimpleNamespace(skill_timers=timers),
            logger=SimpleNamespace(behavior=MagicMock()),
            stop_event=SimpleNamespace(
                wait=stop_wait,
                is_set=gates.is_stopped,
                set=gates.stop_event.set,
            ),
            resume_gate=gates.resume_gate,
            is_stopped=gates.is_stopped,
            should_run_workers=gates.should_run_workers,
            should_run_timers=gates.should_run_timers,
            wait_while_stopped_or_paused=lambda _timeout: gates.should_run_workers(),
        )
        worker = SkillTimerWorker(ctx, SimpleNamespace(teleport_key=teleport_key))

        try:
            with patch(
                "pybot.runtime.workers.skill_timer_worker.monotonic_ms",
                side_effect=lambda: clock["ms"],
            ):
                worker.run()
        finally:
            was_healing = gates.healing_event.is_set()
            gates.end_heal_ops()

        self.assertTrue(was_healing)
        self.assertEqual(presses, [59, 59])
        arm_logs = [
            call.args[0]
            for call in ctx.logger.behavior.call_args_list
            if call.args and "armed" in call.args[0]
        ]
        pause_logs = [
            call.args[0]
            for call in ctx.logger.behavior.call_args_list
            if call.args and "paused" in call.args[0]
        ]
        self.assertEqual(len(arm_logs), 1)
        self.assertEqual(pause_logs, [])

    def test_disarm_on_sit_then_rearm_on_resume(self) -> None:
        timers = (
            SkillTimerRuntime(button="f1", scan_code=59, interval_ms=60_000),
        )
        stop = threading.Event()
        presses: list[int] = []
        clock = {"ms": 0}
        running = {"ok": True}

        def teleport_key(scan_code: int) -> None:
            presses.append((scan_code, clock["ms"]))
            if len(presses) == 1:
                running["ok"] = False
            elif len(presses) >= 2:
                stop.set()

        def should_run() -> bool:
            return running["ok"] and not stop.is_set()

        def wait_paused(timeout_s: float) -> bool:
            # Sit pause tick: resume hunting and advance clock.
            if not running["ok"]:
                running["ok"] = True
                clock["ms"] += 10_000
            else:
                clock["ms"] += int(round(timeout_s * 1000))
            return should_run()

        def stop_wait(timeout_s: float) -> bool:
            if stop.is_set():
                return True
            # Avoid sleeping the real interval in the test.
            clock["ms"] += 1
            return False

        stop_event = SimpleNamespace(wait=stop_wait, is_set=stop.is_set, set=stop.set)
        ctx = SimpleNamespace(
            config=SimpleNamespace(skill_timers=timers),
            logger=SimpleNamespace(behavior=MagicMock()),
            stop_event=stop_event,
            resume_gate=threading.Event(),
            is_stopped=stop.is_set,
            should_run_workers=should_run,
            should_run_timers=should_run,
            wait_while_stopped_or_paused=wait_paused,
        )
        worker = SkillTimerWorker(ctx, SimpleNamespace(teleport_key=teleport_key))

        with patch(
            "pybot.runtime.workers.skill_timer_worker.monotonic_ms",
            side_effect=lambda: clock["ms"],
        ):
            worker.run()

        self.assertEqual([p[0] for p in presses], [59, 59])
        # Second press happens after re-arm (due immediately), not after 60s.
        self.assertLess(presses[1][1] - presses[0][1], 60_000)
        pause_logs = [
            call.args[0]
            for call in ctx.logger.behavior.call_args_list
            if call.args and "paused" in call.args[0]
        ]
        arm_logs = [
            call.args[0]
            for call in ctx.logger.behavior.call_args_list
            if call.args and "armed" in call.args[0]
        ]
        self.assertTrue(pause_logs)
        self.assertGreaterEqual(len(arm_logs), 2)


if __name__ == "__main__":
    unittest.main()
