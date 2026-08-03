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

    def test_stop_refuses_overlap_until_runtime_cleanup_completes(self) -> None:
        app_config = MagicMock()
        app_config.window_id = 1
        app_config.hunt_validation_log = False
        controller = BotController(
            app_config=app_config,
            session_id="test_pending_shutdown",
        )
        cleanup_complete = {"value": False}
        retry_calls = {"n": 0}
        runtime = MagicMock()

        def run(**_kwargs) -> int:
            return 0

        def retry_shutdown() -> bool:
            retry_calls["n"] += 1
            if retry_calls["n"] >= 2:
                cleanup_complete["value"] = True
            return cleanup_complete["value"]

        runtime.run = run
        runtime.stop = MagicMock()
        runtime.is_shutdown_complete.side_effect = (
            lambda: cleanup_complete["value"]
        )
        runtime.retry_shutdown.side_effect = retry_shutdown

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
                thread = controller._thread
                self.assertIsNotNone(thread)
                assert thread is not None
                thread.join(timeout=1.0)
                self.assertFalse(thread.is_alive())
                self.assertTrue(controller.shutdown_pending)

                self.assertFalse(controller.stop(join_timeout=1.0))
                self.assertTrue(controller.shutdown_pending)
                # A retained incomplete runtime cannot be replaced.
                controller.start(mob_name="new-horn")
                self.assertIs(controller._runtime, runtime)

                self.assertTrue(controller.stop(join_timeout=1.0))
                self.assertFalse(controller.shutdown_pending)
                self.assertIsNone(controller._runtime)

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


class BotLifecycleStartingCleanupTests(unittest.TestCase):
    def test_start_thread_creation_failure_does_not_strand_starting(self) -> None:
        root = MagicMock()
        root.after = MagicMock()
        lifecycle = BotLifecycleManager(
            root=root,
            config=MagicMock(),
            mob_catalog=[],
            session=MagicMock(),
            viiper=MagicMock(),
        )

        class StartFailThread:
            def __init__(self, **_kwargs) -> None:
                pass

            def start(self) -> None:
                raise RuntimeError("thread creation failed")

        with patch("pybot.app.bot_lifecycle.threading.Thread", StartFailThread):
            with self.assertRaisesRegex(RuntimeError, "thread creation failed"):
                lifecycle.start(
                    config_snapshot=MagicMock(),
                    session_id="thread-failure",
                )

        self.assertEqual(lifecycle.state, BotState.OFF)
        self.assertFalse(lifecycle.stopping)
        self.assertTrue(lifecycle.shutdown_ready)

    def test_start_failure_after_controller_creation_keeps_cleanup_owned(self) -> None:
        root = MagicMock()
        root.after = MagicMock()
        session = MagicMock()
        lifecycle = BotLifecycleManager(
            root=root,
            config=MagicMock(),
            mob_catalog=[MagicMock()],
            session=session,
            viiper=MagicMock(),
        )
        bot = MagicMock()
        bot.running = True
        bot.shutdown_pending = True
        release_stop = threading.Event()

        def stop_after_release(**_kwargs) -> bool:
            release_stop.wait(timeout=2.0)
            return True

        bot.stop.side_effect = stop_after_release

        with (
            patch("pybot.app.bot_lifecycle.restore_and_activate"),
            patch("pybot.app.bot_lifecycle.mob_folder_by_index", return_value="wolf"),
            patch("pybot.app.bot_lifecycle.BotController", return_value=bot),
            patch.object(bot, "start", side_effect=RuntimeError("thread launch failed")),
        ):
            cfg = MagicMock(
                window_id=1,
                selected_monster=0,
                hunt_log_overlay=False,
            )
            self.assertTrue(lifecycle.start(config_snapshot=cfg, session_id="failed-start"))
            start_thread = lifecycle._start_thread
            self.assertIsNotNone(start_thread)
            assert start_thread is not None
            start_thread.join(timeout=2.0)
            self.assertFalse(start_thread.is_alive())

        self.assertTrue(lifecycle.stopping)
        self.assertIs(lifecycle._bot, bot)
        self.assertEqual(lifecycle.state, BotState.STOPPING)
        bot.request_stop.assert_called_once_with()
        session.end.assert_not_called()

        joiner = lifecycle._stop_joiner
        self.assertIsNotNone(joiner)
        assert joiner is not None
        release_stop.set()
        joiner.join(timeout=2.0)
        self.assertFalse(joiner.is_alive())
        lifecycle._refresh_stopped_state()
        self.assertEqual(lifecycle.state, BotState.OFF)
        self.assertFalse(lifecycle.stopping)
        session.end.assert_called_once_with("bot start cancelled")


class BotLifecycleOrphanCleanupTests(unittest.TestCase):
    def test_orphan_cleanup_retry_is_tracked_by_shutdown_readiness(self) -> None:
        root = MagicMock()
        root.after = MagicMock()
        lifecycle = BotLifecycleManager(
            root=root,
            config=MagicMock(),
            mob_catalog=[],
            session=MagicMock(),
            viiper=MagicMock(),
        )
        bot = MagicMock()
        bot.shutdown_pending = True
        outcomes = iter([False, False, False, True])
        first_attempt = threading.Event()

        def stop_with_signal(**_kwargs) -> bool:
            first_attempt.set()
            return next(outcomes)

        bot.stop.side_effect = stop_with_signal

        lifecycle._start_orphan_stop_joiner(bot, end_session=False)
        self.assertTrue(first_attempt.wait(timeout=2.0))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if bot.stop.call_count >= 3 and not lifecycle._orphan_cleanup_threads:
                break
            time.sleep(0.01)
        self.assertGreaterEqual(bot.stop.call_count, 3)
        self.assertFalse(lifecycle.shutdown_ready)

        lifecycle.retry_shutdown_cleanup()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if bot.stop.call_count >= 4 and lifecycle.shutdown_ready:
                break
            time.sleep(0.01)
        self.assertGreaterEqual(bot.stop.call_count, 4)
        self.assertTrue(lifecycle.shutdown_ready)


class BotLifecycleStoppingTests(unittest.TestCase):
    def test_stop_owns_shutdown_and_refuses_restart_until_bot_exits(self) -> None:
        root = MagicMock()
        root.after = MagicMock()
        lifecycle = BotLifecycleManager(
            root=root,
            config=MagicMock(),
            mob_catalog=[],
            session=MagicMock(),
            viiper=MagicMock(),
            hunt_overlay=MagicMock(),
        )
        bot = MagicMock()
        stop_called = threading.Event()
        release_stop = threading.Event()

        def stop(join_timeout: float = 3.0) -> bool:
            del join_timeout
            stop_called.set()
            release_stop.wait(timeout=2.0)
            return True

        bot.stop.side_effect = stop
        lifecycle._bot = bot
        lifecycle._state = BotState.RUNNING

        lifecycle.stop()
        self.assertTrue(stop_called.wait(timeout=1.0))

        self.assertEqual(lifecycle.state, BotState.STOPPING)
        self.assertTrue(lifecycle.stopping)
        lifecycle.stop()
        bot.request_stop.assert_called_once()
        self.assertFalse(
            lifecycle.start(config_snapshot=MagicMock(), session_id="new-session")
        )

        release_stop.set()
        joiner = lifecycle._stop_joiner
        self.assertIsNotNone(joiner)
        assert joiner is not None
        joiner.join(timeout=2.0)
        self.assertFalse(joiner.is_alive())
        lifecycle._refresh_stopped_state()
        self.assertEqual(lifecycle.state, BotState.OFF)
        self.assertFalse(lifecycle.stopping)


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

            # Do not overlap the cancelled startup thread. Restart becomes
            # available only after that thread has observed cancellation.
            self.assertFalse(lifecycle.start(config_snapshot=cfg, session_id="s1"))
            release.set()

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                root.update()
                if (
                    lifecycle._start_thread is not None
                    and not lifecycle._start_thread.is_alive()
                ):
                    break
                time.sleep(0.02)
            self.assertTrue(
                lifecycle._start_thread is not None
                and not lifecycle._start_thread.is_alive()
            )
            self.assertTrue(lifecycle.start(config_snapshot=cfg, session_id="s1"))
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and lifecycle.state != BotState.RUNNING:
                root.update()
                time.sleep(0.02)
            self.assertEqual(lifecycle.state, BotState.RUNNING)


if __name__ == "__main__":
    unittest.main()
