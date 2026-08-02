"""Sit/hunt gate races: user pause must not drop sit; one worker set only."""

from __future__ import annotations

import ctypes
import threading
import unittest
from unittest.mock import MagicMock

from pybot.runtime.gate_controller import GateController

from pybot.runtime.runtime_context import HuntRuntimeContext

# HuntRuntime pulls in the Windows Viiper backend at import time.
if not hasattr(ctypes, "windll"):
    ctypes.windll = MagicMock()  # type: ignore[attr-defined]
    ctypes.windll.user32 = MagicMock()

from pybot.runtime.hunt_runtime import HuntRuntime  # noqa: E402


class SitHuntGateRaceTests(unittest.TestCase):
    def test_wait_while_user_paused_ignores_sitting_gate(self) -> None:
        ctx = HuntRuntimeContext(
            config=MagicMock(),
            logger=MagicMock(),
            tracks=MagicMock(),
            policy=MagicMock(),
            capture=MagicMock(),
            detector=MagicMock(),
            tracker=MagicMock(),
            validation=MagicMock(),
            control=MagicMock(),
        )
        ctx.begin_sit_ops()
        ctx.pause_event.set()
        # Sitting alone must not make this wait fail after timeout.
        self.assertFalse(ctx.should_run_workers())
        self.assertFalse(ctx.wait_while_user_paused(0.05))
        ctx.pause_event.clear()
        self.assertTrue(ctx.wait_while_user_paused(0.05))
        self.assertFalse(ctx.should_run_workers())

    def test_input_admission_serializes_sit_claim(self) -> None:
        gates = GateController()
        action_started = threading.Event()
        release_action = threading.Event()
        sit_claimed = threading.Event()
        result: list[bool] = []

        def action() -> bool:
            action_started.set()
            release_action.wait(1.0)
            return True

        input_thread = threading.Thread(
            target=lambda: gates.perform_input_if_allowed(lambda: True, action),
            daemon=True,
        )
        input_thread.start()
        self.assertTrue(action_started.wait(1.0))

        def claim_sit() -> None:
            result.append(gates.try_begin_sit_ops())
            sit_claimed.set()

        sit_thread = threading.Thread(target=claim_sit, daemon=True)
        sit_thread.start()
        self.assertFalse(sit_claimed.wait(0.05))

        release_action.set()
        input_thread.join(timeout=1.0)
        sit_thread.join(timeout=1.0)

        self.assertFalse(input_thread.is_alive())
        self.assertFalse(sit_thread.is_alive())
        self.assertEqual(result, [True])
        self.assertTrue(gates.sitting_event.is_set())

    def test_run_refuses_when_worker_threads_still_alive(self) -> None:
        ctx = MagicMock()
        runtime = HuntRuntime.__new__(HuntRuntime)
        runtime._ctx = ctx
        zombie = MagicMock()
        zombie.name = "AttackLoop"
        zombie.is_alive.return_value = True
        runtime._worker_threads = [zombie]
        with self.assertRaises(RuntimeError) as raised:
            runtime.run()
        self.assertIn("only one worker set may run", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
