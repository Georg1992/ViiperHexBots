"""stop → start → stop → start must not overlap hunt runtimes."""

from __future__ import annotations

import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from pybot.app.bot_controller import BotController
from pybot.runtime.hunt_runtime import HuntRuntime, RuntimeDependencies


def _build_runtime() -> tuple[HuntRuntime, threading.Event, MagicMock, list[str]]:
    stop_event = threading.Event()
    discovery_wake = threading.Event()
    phases: list[str] = []

    def worker() -> None:
        phases.append("start")
        while not stop_event.is_set():
            discovery_wake.wait(0.02)
            discovery_wake.clear()
        phases.append("exit")

    ctx = MagicMock()
    ctx.is_stopped.side_effect = stop_event.is_set
    ctx.stop_event = stop_event
    ctx.discovery_wake = discovery_wake
    ctx.pause_event = threading.Event()
    ctx.sitting_event = threading.Event()
    ctx.control.poll.return_value = None
    ctx.config.mob_name = "horn"
    ctx.config.hwnd = 0
    ctx.config.hunt_mode = "teleport"
    ctx.config.skill_button = "e"
    ctx.config.teleport_button = "q"
    ctx.capture.get_hunt_roi.return_value = None
    ctx.logger.behavior = MagicMock()
    ctx.mark_running = MagicMock()
    ctx.mark_paused = MagicMock()

    input_backend = MagicMock()
    deps = RuntimeDependencies(
        ctx=ctx,
        input_backend=input_backend,
        hunt_mode=MagicMock(),
        logger=ctx.logger,
        workers=[("worker", worker)],
    )
    return HuntRuntime(deps), stop_event, input_backend, phases


class HuntRuntimeStopStartCycleTests(unittest.TestCase):
    def test_stop_start_stop_start_shuts_down_each_generation(self) -> None:
        for _cycle in range(2):
            runtime, _stop_event, input_backend, phases = _build_runtime()
            thread = threading.Thread(target=runtime.run, daemon=True)
            thread.start()

            for _ in range(100):
                if "start" in phases:
                    break
                threading.Event().wait(0.02)
            self.assertIn("start", phases)

            runtime.stop()
            thread.join(timeout=5.0)
            self.assertFalse(thread.is_alive())
            self.assertIn("exit", phases)
            input_backend.shutdown.assert_called()


class BotControllerStopStartCycleTests(unittest.TestCase):
    def test_stop_clears_handles_only_after_thread_exits(self) -> None:
        app_config = MagicMock()
        app_config.window_id = 1
        app_config.hunt_validation_log = False
        controller = BotController(
            app_config=app_config,
            session_id="test_stop_start",
        )

        release = threading.Event()
        started = threading.Event()

        def slow_run(**_kwargs) -> int:
            started.set()
            release.wait(timeout=5.0)
            return 0

        runtime = MagicMock()
        runtime.run = slow_run
        # stop() must stay non-blocking (matches HuntRuntime.stop).
        runtime.stop = MagicMock()

        with TemporaryDirectory() as tmp:
            sessions = Path(tmp)
            with patch(
                "pybot.app.bot_controller.create_runtime_deps",
                return_value=MagicMock(),
            ), patch(
                "pybot.app.bot_controller.load_runtime_config",
                return_value=MagicMock(),
            ), patch(
                "pybot.app.bot_controller.HuntRuntime",
                return_value=runtime,
            ), patch(
                "pybot.app.bot_controller.SESSIONS_DIR",
                sessions,
            ):
                controller.start(mob_name="horn")
                self.assertTrue(started.wait(timeout=2.0))
                self.assertTrue(controller.running)

                stopped = controller.stop(join_timeout=0.05)
                self.assertFalse(stopped)
                self.assertTrue(controller.running)

                release.set()
                stopped = controller.stop(join_timeout=2.0)
                self.assertTrue(stopped)
                self.assertFalse(controller.running)

    def test_stop_start_stop_start_cycle(self) -> None:
        app_config = MagicMock()
        app_config.window_id = 1
        app_config.hunt_validation_log = False
        controller = BotController(
            app_config=app_config,
            session_id="test_cycle",
        )
        run_count = {"n": 0}

        def make_runtime(_deps) -> MagicMock:
            stop_flag = threading.Event()
            rt = MagicMock()

            def run(**_kwargs) -> int:
                run_count["n"] += 1
                while not stop_flag.is_set():
                    stop_flag.wait(0.02)
                return 0

            rt.run = run
            rt.stop = MagicMock(side_effect=stop_flag.set)
            return rt

        with TemporaryDirectory() as tmp:
            sessions = Path(tmp)
            with patch(
                "pybot.app.bot_controller.create_runtime_deps",
                return_value=MagicMock(),
            ), patch(
                "pybot.app.bot_controller.load_runtime_config",
                return_value=MagicMock(),
            ), patch(
                "pybot.app.bot_controller.HuntRuntime",
                side_effect=make_runtime,
            ), patch(
                "pybot.app.bot_controller.SESSIONS_DIR",
                sessions,
            ):
                for _ in range(2):
                    controller.start(mob_name="horn")
                    self.assertTrue(controller.running)
                    self.assertTrue(controller.stop(join_timeout=2.0))
                    self.assertFalse(controller.running)

        self.assertEqual(run_count["n"], 2)


if __name__ == "__main__":
    unittest.main()
