"""stop → start → stop → start must not overlap hunt runtimes."""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import tkinter as tk

from pybot.app.bot_controller import BotController
from pybot.app.bot_lifecycle import BotLifecycleManager, BotState
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
    ctx.resume_gate = threading.Event()
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


class HuntRuntimeStopWakesPausedWorkersTests(unittest.TestCase):
    def test_stop_sets_resume_gate_so_paused_workers_can_exit(self) -> None:
        runtime, stop_event, _input_backend, _phases = _build_runtime()
        runtime._ctx.resume_gate.clear()
        runtime._ctx.pause_event.set()
        self.assertFalse(runtime._ctx.resume_gate.is_set())

        runtime.stop()

        self.assertTrue(stop_event.is_set())
        self.assertTrue(runtime._ctx.resume_gate.is_set())


class CaptureSessionResetTests(unittest.TestCase):
    def test_reset_rotates_lock_when_previous_holder_is_stuck(self) -> None:
        import pybot.recognition.capture as capture

        stuck = threading.Lock()
        stuck.acquire()
        capture._capture_lock = stuck
        capture._sct = None

        capture.reset_capture_session()

        self.assertIsNot(capture._capture_lock, stuck)
        self.assertTrue(capture._capture_lock.acquire(timeout=0.1))
        capture._capture_lock.release()
        stuck.release()


class BotLifecycleRestartAfterCancelTests(unittest.TestCase):
    def test_restart_accepted_while_cancelled_start_thread_still_alive(self) -> None:
        """Stop during STARTING must not permanently block the next Start."""
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)

        cfg = MagicMock()
        cfg.window_id = 1
        cfg.selected_monster = 0
        cfg.hunt_log_overlay = False
        cfg.search_range = 5

        lifecycle = BotLifecycleManager(
            root=root,
            config=cfg,
            mob_catalog=[MagicMock()],
            session=MagicMock(),
            viiper=MagicMock(),
            hunt_overlay=MagicMock(),
        )

        started = threading.Event()
        release = threading.Event()

        class SlowStartBot:
            def __init__(self, *a, **k):
                self._alive = False

            @property
            def running(self) -> bool:
                return self._alive

            def start(self, **k) -> None:
                started.set()
                release.wait(timeout=5.0)
                self._alive = True

            def request_stop(self) -> None:
                pass

            def stop(self, join_timeout: float = 3.0) -> bool:
                self._alive = False
                return True

            def pause(self) -> None:
                pass

            def resume(self) -> None:
                pass

            def set_search_range_cells(self, _cells: int) -> None:
                pass

        with (
            patch("pybot.app.bot_lifecycle.restore_and_activate"),
            patch("pybot.app.bot_lifecycle.mob_folder_by_index", return_value="wolf"),
            patch("pybot.app.bot_lifecycle.BotController", SlowStartBot),
            patch("pybot.app.bot_lifecycle.NullOverlay"),
        ):
            self.assertTrue(lifecycle.start(config_snapshot=cfg, session_id="s1"))
            self.assertTrue(started.wait(timeout=2.0))
            lifecycle.stop()
            self.assertEqual(lifecycle.state, BotState.OFF)
            self.assertTrue(
                lifecycle._start_thread is not None
                and lifecycle._start_thread.is_alive()
            )

            self.assertTrue(lifecycle.start(config_snapshot=cfg, session_id="s1"))
            self.assertEqual(lifecycle.state, BotState.STARTING)
            release.set()

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and lifecycle.state != BotState.RUNNING:
                root.update()
                time.sleep(0.02)
            self.assertEqual(lifecycle.state, BotState.RUNNING)


if __name__ == "__main__":
    unittest.main()
