"""Hunt runtime must fully stop workers before a restart."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import MagicMock

from pybot.runtime.hunt_runtime import HuntRuntime, RuntimeDependencies


class HuntRuntimeShutdownTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
