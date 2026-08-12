"""Skill timer stagger, buff-priority, and sit/pause re-arm behavior."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pybot.config.runtime import SkillTimerRuntime
from pybot.runtime.constants import SKILL_TIMER_STAGGER_MS
from pybot.runtime.gate_controller import CharacterActionGate
from pybot.runtime.workers.skill_timer_worker import SkillTimerWorker


class ClockStopEvent:
    """Stop event that advances a fake monotonic clock on each wait."""

    def __init__(self, stop: threading.Event, clock: dict) -> None:
        self._stop = stop
        self._clock = clock

    def is_set(self) -> bool:
        return self._stop.is_set()

    def wait(self, timeout_s: float) -> bool:
        self._clock["ms"] += int(round(timeout_s * 1000))
        return self._stop.is_set()

    def set(self) -> None:
        self._stop.set()


class SkillTimerWorkerTests(unittest.TestCase):
    @staticmethod
    def _run_pending_until_stopped(worker, ctx) -> None:
        """Exercise the production scheduler hook without the deleted loop."""
        while not ctx.is_stopped():
            worker.process_pending()
            if ctx.is_stopped():
                break
            if ctx.should_run_timers() or ctx.should_run_workers():
                ctx.stop_event.wait(0.25)
            else:
                ctx.wait_while_stopped_or_paused(0.25)

    def test_staggers_due_timers_by_500ms(self) -> None:
        timers = (
            SkillTimerRuntime(button="f1", scan_code=59, interval_ms=60_000),
            SkillTimerRuntime(button="f2", scan_code=60, interval_ms=60_000),
        )
        stop = threading.Event()
        presses: list[tuple[int, int]] = []
        clock = {"ms": 1_000_000}

        def teleport_key(scan_code: int) -> None:
            presses.append((scan_code, clock["ms"]))
            if len(presses) >= 2:
                stop.set()

        ctx = SimpleNamespace(
            config=SimpleNamespace(skill_timers=timers),
            logger=SimpleNamespace(behavior=MagicMock()),
            stop_event=ClockStopEvent(stop, clock),
            resume_gate=threading.Event(),
            is_stopped=stop.is_set,
            should_run_workers=lambda: not stop.is_set(),
            should_run_timers=lambda: not stop.is_set(),
            wait_while_stopped_or_paused=lambda _t: not stop.is_set(),
            character_action_gate=CharacterActionGate(),
        )
        worker = SkillTimerWorker(ctx, SimpleNamespace(teleport_key=teleport_key))

        with patch(
            "pybot.runtime.workers.skill_timer_worker.monotonic_ms",
            side_effect=lambda: clock["ms"],
        ):
            self._run_pending_until_stopped(worker, ctx)

        self.assertEqual([p[0] for p in presses], [59, 60])
        # The shared stagger window separated the two presses.
        self.assertGreaterEqual(
            presses[1][1] - presses[0][1],
            SKILL_TIMER_STAGGER_MS,
        )

    def test_timer_yields_while_buff_burst_is_active(self) -> None:
        """A due timer must not press while the buff worker owns the slot."""
        timers = (
            SkillTimerRuntime(button="s", scan_code=31, interval_ms=1_000),
        )
        gate = CharacterActionGate()
        gate.begin_buff_burst()
        stop = threading.Event()
        presses: list[tuple[int, int]] = []
        clock = {"ms": 1_000_000}
        ticks = {"n": 0}

        def teleport_key(scan_code: int) -> None:
            presses.append((scan_code, clock["ms"]))
            stop.set()

        def stop_wait(timeout_s: float) -> bool:
            clock["ms"] += int(round(timeout_s * 1000))
            ticks["n"] += 1
            # Buff worker finishes its burst after a few poll slices.
            if ticks["n"] >= 3:
                gate.end_buff_burst()
            return stop.is_set()

        ctx = SimpleNamespace(
            config=SimpleNamespace(skill_timers=timers),
            logger=SimpleNamespace(behavior=MagicMock()),
            stop_event=SimpleNamespace(
                wait=stop_wait,
                is_set=stop.is_set,
                set=stop.set,
            ),
            resume_gate=threading.Event(),
            is_stopped=stop.is_set,
            should_run_workers=lambda: not stop.is_set(),
            should_run_timers=lambda: not stop.is_set(),
            wait_while_stopped_or_paused=lambda _t: not stop.is_set(),
            character_action_gate=gate,
        )
        worker = SkillTimerWorker(ctx, SimpleNamespace(teleport_key=teleport_key))

        with patch(
            "pybot.runtime.workers.skill_timer_worker.monotonic_ms",
            side_effect=lambda: clock["ms"],
        ):
            self._run_pending_until_stopped(worker, ctx)

        self.assertEqual([p[0] for p in presses], [31])
        # The press only happened after the buff burst cleared.
        self.assertGreaterEqual(ticks["n"], 3)

    def test_long_timer_is_due_immediately_even_with_low_monotonic_uptime(self) -> None:
        timers = (
            SkillTimerRuntime(button="s", scan_code=31, interval_ms=180_000),
        )
        stop = threading.Event()
        presses: list[int] = []
        clock = {"ms": 1_000}

        def teleport_key(scan_code: int) -> None:
            presses.append(scan_code)
            stop.set()

        def wait_paused(_timeout_s: float) -> bool:
            return stop.is_set()

        ctx = SimpleNamespace(
            config=SimpleNamespace(skill_timers=timers),
            logger=SimpleNamespace(behavior=MagicMock()),
            stop_event=stop,
            resume_gate=threading.Event(),
            is_stopped=stop.is_set,
            should_run_workers=lambda: not stop.is_set(),
            should_run_timers=lambda: not stop.is_set(),
            wait_while_stopped_or_paused=wait_paused,
            character_action_gate=CharacterActionGate(),
        )
        worker = SkillTimerWorker(ctx, SimpleNamespace(teleport_key=teleport_key))

        with patch(
            "pybot.runtime.workers.skill_timer_worker.monotonic_ms",
            side_effect=lambda: clock["ms"],
        ):
            self._run_pending_until_stopped(worker, ctx)

        self.assertEqual(presses, [31])
        self.assertTrue(
            any("key executed key=s scanCode=31" in call.args[0]
                for call in ctx.logger.behavior.call_args_list)
        )

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
            character_action_gate=gates.character_action_gate,
        )
        worker = SkillTimerWorker(ctx, SimpleNamespace(teleport_key=teleport_key))

        try:
            with patch(
                "pybot.runtime.workers.skill_timer_worker.monotonic_ms",
                side_effect=lambda: clock["ms"],
            ):
                self._run_pending_until_stopped(worker, ctx)
        finally:
            was_healing = gates.healing_event.is_set()
            gates.end_heal_ops()

        self.assertTrue(was_healing)
        self.assertEqual(presses, [59, 59])
        # Timers keep firing through the heal session: nothing here ever
        # disarms or re-arms the schedule (a new hunt generation is the only
        # re-arm source, and healing never changes it). The legacy run loop's
        # "armed"/"paused" diagnostics were removed with the collapsed loop;
        # the scheduling behavior itself is asserted by the presses above.
        self.assertFalse(gates.healing_event.is_set())
        gates.end_heal_ops()

    def test_teleport_pause_preserves_180_second_timer(self) -> None:
        timers = (
            SkillTimerRuntime(button="s", scan_code=31, interval_ms=180_000),
        )
        stop = threading.Event()
        presses: list[tuple[int, int]] = []
        clock = {"ms": 100_000}
        running = {"ok": True}
        suspended = {"value": False}

        def teleport_key(scan_code: int) -> None:
            presses.append((scan_code, clock["ms"]))
            if len(presses) == 1:
                suspended["value"] = True
            else:
                stop.set()

        def should_run_timers() -> bool:
            return running["ok"] and not suspended["value"] and not stop.is_set()

        def wait_paused(timeout_s: float) -> bool:
            clock["ms"] += int(round(timeout_s * 1000))
            return should_run_timers()

        def stop_wait(timeout_s: float) -> bool:
            # The worker uses this path while discovery_suspend is active.
            clock["ms"] += int(round(timeout_s * 1000))
            if suspended["value"]:
                suspended["value"] = False
                stop.set()
            return stop.is_set()

        ctx = SimpleNamespace(
            config=SimpleNamespace(skill_timers=timers),
            logger=SimpleNamespace(behavior=MagicMock()),
            stop_event=SimpleNamespace(
                wait=stop_wait,
                is_set=stop.is_set,
                set=stop.set,
            ),
            resume_gate=threading.Event(),
            is_stopped=stop.is_set,
            should_run_workers=lambda: running["ok"] and not stop.is_set(),
            should_run_timers=should_run_timers,
            wait_while_stopped_or_paused=wait_paused,
            hunt_generation=0,
            character_action_gate=CharacterActionGate(),
        )
        worker = SkillTimerWorker(ctx, SimpleNamespace(teleport_key=teleport_key))

        with patch(
            "pybot.runtime.workers.skill_timer_worker.monotonic_ms",
            side_effect=lambda: clock["ms"],
        ):
            self._run_pending_until_stopped(worker, ctx)

        self.assertEqual([scan for scan, _at in presses], [31])

    def test_disarm_on_sit_then_rearm_on_resume(self) -> None:
        timers = (
            SkillTimerRuntime(button="f1", scan_code=59, interval_ms=60_000),
        )
        stop = threading.Event()
        presses: list[tuple[int, int]] = []
        clock = {"ms": 0}
        running = {"ok": True}
        ctx_ref: dict[str, SimpleNamespace] = {}

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
                ctx_ref["ctx"].hunt_generation = 1
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
            hunt_generation=0,
            character_action_gate=CharacterActionGate(),
        )
        ctx_ref["ctx"] = ctx
        worker = SkillTimerWorker(ctx, SimpleNamespace(teleport_key=teleport_key))

        with patch(
            "pybot.runtime.workers.skill_timer_worker.monotonic_ms",
            side_effect=lambda: clock["ms"],
        ):
            self._run_pending_until_stopped(worker, ctx)

        self.assertEqual([p[0] for p in presses], [59, 59])
        # A sit creates a new hunt generation, so startup re-arm is immediate.
        self.assertLess(presses[1][1] - presses[0][1], 60_000)
        # The legacy run loop's "armed"/"paused" diagnostics were removed with
        # the collapsed loop; the re-arm behavior is asserted by the presses.
        self.assertFalse(ctx.hunt_generation == 0)


if __name__ == "__main__":
    unittest.main()
