"""Hunt runtime must fully stop workers before a restart."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import MagicMock, patch

from pybot.runtime.hunt_runtime import HuntRuntime, RuntimeDependencies


class HuntRuntimeShutdownTests(unittest.TestCase):
    def test_incomplete_worker_shutdown_keeps_runtime_owned_for_retry(self) -> None:
        stop_event = threading.Event()
        discovery_wake = threading.Event()
        worker_started = threading.Event()
        release_worker = threading.Event()

        def stuck_worker() -> None:
            worker_started.set()
            release_worker.wait(timeout=5.0)

        ctx = MagicMock()
        ctx.is_stopped.side_effect = stop_event.is_set
        ctx.stop_event = stop_event
        ctx.discovery_wake = discovery_wake
        ctx.resume_gate = threading.Event()
        ctx.control.poll.return_value = None
        ctx.config.mob_name = "horn"
        ctx.config.hwnd = 0
        ctx.config.hunt_mode = "teleport"
        ctx.config.skill_button = "e"
        ctx.config.teleport_button = "q"
        ctx.capture.get_hunt_roi.return_value = None
        ctx.logger.behavior = MagicMock()

        input_backend = MagicMock()
        runtime = HuntRuntime(
            RuntimeDependencies(
                ctx=ctx,
                input_backend=input_backend,
                hunt_mode=MagicMock(),
                logger=ctx.logger,
                workers=[("stuck", stuck_worker)],
            )
        )

        with patch("pybot.runtime.hunt_runtime.WORKER_SHUTDOWN_TIMEOUT_S", 0.05):
            thread = threading.Thread(target=runtime.run, daemon=True)
            thread.start()
            self.assertTrue(worker_started.wait(timeout=2.0))
            runtime.stop()
            thread.join(timeout=2.0)

            self.assertFalse(thread.is_alive())
            self.assertFalse(runtime.is_shutdown_complete())
            input_backend.shutdown.assert_not_called()

            release_worker.set()
            self.assertTrue(runtime.retry_shutdown())
            self.assertTrue(runtime.is_shutdown_complete())
            input_backend.shutdown.assert_called_once()
            ctx.logger.close.assert_called_once()

    def test_retry_shutdown_waits_for_logger_cleanup(self) -> None:
        ctx = MagicMock()
        ctx.stop_event = threading.Event()
        ctx.discovery_wake = threading.Event()
        ctx.resume_gate = threading.Event()
        ctx.logger.behavior = MagicMock()
        logger = MagicMock()
        logger.close.side_effect = [False, True]

        input_backend = MagicMock()
        runtime = HuntRuntime(
            RuntimeDependencies(
                ctx=ctx,
                input_backend=input_backend,
                hunt_mode=MagicMock(),
                logger=logger,
                workers=[],
            )
        )

        self.assertFalse(runtime.retry_shutdown())
        self.assertFalse(runtime.is_shutdown_complete())
        self.assertTrue(runtime.retry_shutdown())
        self.assertTrue(runtime.is_shutdown_complete())
        self.assertEqual(logger.close.call_count, 2)

    def test_unresolved_sit_cleanup_blocks_shutdown_until_retry_succeeds(self) -> None:
        ctx = MagicMock()
        ctx.stop_event = threading.Event()
        ctx.discovery_wake = threading.Event()
        ctx.resume_gate = threading.Event()
        ctx.sit_cleanup_unresolved = threading.Event()
        ctx.sit_cleanup_unresolved.set()
        ctx.retry_sit_cleanup.side_effect = [False, True]
        ctx.logger.behavior = MagicMock()

        input_backend = MagicMock()
        runtime = HuntRuntime(
            RuntimeDependencies(
                ctx=ctx,
                input_backend=input_backend,
                hunt_mode=MagicMock(),
                logger=ctx.logger,
                workers=[],
            )
        )

        self.assertFalse(runtime.retry_shutdown())
        self.assertFalse(runtime.is_shutdown_complete())
        input_backend.shutdown.assert_not_called()

        self.assertTrue(runtime.retry_shutdown())
        self.assertTrue(runtime.is_shutdown_complete())
        input_backend.shutdown.assert_called_once()
        self.assertEqual(ctx.retry_sit_cleanup.call_count, 2)

    def test_run_joins_workers_and_shuts_down_input(self) -> None:
        stop_event = threading.Event()
        discovery_wake = threading.Event()
        worker_started = threading.Event()
        worker_exited = threading.Event()

        def worker() -> None:
            worker_started.set()
            while not stop_event.is_set():
                discovery_wake.wait(0.05)
            worker_exited.set()

        ctx = MagicMock()
        ctx.is_stopped.side_effect = stop_event.is_set
        ctx.stop_event = stop_event
        ctx.discovery_wake = discovery_wake
        ctx.pause_event = threading.Event()
        ctx.control.poll.return_value = None
        ctx.config.mob_name = "horn"
        ctx.config.hwnd = 0
        ctx.config.hunt_mode = "teleport"
        ctx.config.skill_button = "e"
        ctx.config.teleport_button = "q"
        ctx.capture.get_hunt_roi.return_value = None
        ctx.logger.behavior = MagicMock()

        input_backend = MagicMock()
        deps = RuntimeDependencies(
            ctx=ctx,
            input_backend=input_backend,
            hunt_mode=MagicMock(),
            logger=ctx.logger,
            workers=[("worker", worker)],
        )
        runtime = HuntRuntime(deps)

        thread = threading.Thread(target=runtime.run, daemon=True)
        thread.start()

        self.assertTrue(worker_started.wait(timeout=2.0))
        runtime.stop()
        thread.join(timeout=5.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(worker_exited.is_set())
        input_backend.shutdown.assert_called_once()
        ctx.logger.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
