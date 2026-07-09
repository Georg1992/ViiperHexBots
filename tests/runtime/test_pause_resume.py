"""Pause/resume behavior for hunt runtime and lifecycle."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock

from pybot.app.bot_lifecycle import BotLifecycleManager, BotState
from pybot.runtime.hunt_runtime import HuntRuntime
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.workers.discovery_worker import DiscoveryWorker


class HuntRuntimeContextPauseTests(unittest.TestCase):
    def test_should_run_workers_respects_pause(self) -> None:
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
        self.assertTrue(ctx.should_run_workers())
        ctx.pause_event.set()
        self.assertFalse(ctx.should_run_workers())

    def test_wait_while_stopped_or_paused_returns_when_resumed(self) -> None:
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
        ctx.pause_event.set()

        def resume_soon() -> None:
            time.sleep(0.05)
            ctx.pause_event.clear()

        threading.Thread(target=resume_soon, daemon=True).start()
        self.assertTrue(ctx.wait_while_stopped_or_paused(1.0))


class HuntRuntimePauseTests(unittest.TestCase):
    def test_resume_wakes_discovery(self) -> None:
        ctx = MagicMock()
        ctx.pause_event = threading.Event()
        ctx.discovery_wake = threading.Event()
        ctx.resume_gate = threading.Event()
        ctx.logger = MagicMock()

        def mark_paused() -> None:
            ctx.pause_event.set()
            ctx.resume_gate.clear()

        def mark_running() -> None:
            ctx.pause_event.clear()
            ctx.resume_gate.set()
            ctx.discovery_wake.set()

        ctx.mark_paused = mark_paused
        ctx.mark_running = mark_running

        runtime = HuntRuntime.__new__(HuntRuntime)
        runtime._ctx = ctx

        runtime.pause()
        self.assertTrue(ctx.pause_event.is_set())

        runtime.resume()
        self.assertFalse(ctx.pause_event.is_set())
        self.assertTrue(ctx.discovery_wake.is_set())


class DiscoveryWorkerPauseTests(unittest.TestCase):
    def test_paused_worker_does_not_busy_loop_on_pending_wake(self) -> None:
        ctx = MagicMock()
        ctx.stop_event = threading.Event()
        ctx.pause_event = threading.Event()
        ctx.discovery_wake = threading.Event()
        ctx.discovery_wake.set()
        ctx.resume_gate = threading.Event()
        ctx.should_run_workers.return_value = False
        ctx.config.discovery_interval_ms = 1000
        ctx.logger = MagicMock()

        def wait_while_stopped_or_paused(timeout_s: float) -> bool:
            deadline = time.monotonic() + timeout_s
            while not ctx.stop_event.is_set():
                if ctx.should_run_workers():
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return ctx.should_run_workers()
                if ctx.resume_gate.wait(min(0.025, remaining)):
                    return True
            return False

        ctx.wait_while_stopped_or_paused = wait_while_stopped_or_paused

        worker = DiscoveryWorker(ctx, MagicMock())

        def stop_after_delay() -> None:
            time.sleep(0.15)
            ctx.stop_event.set()

        threading.Thread(target=stop_after_delay, daemon=True).start()
        start = time.monotonic()
        worker.run()
        elapsed = time.monotonic() - start

        self.assertGreaterEqual(elapsed, 0.1)
        self.assertTrue(ctx.discovery_wake.is_set())


class BotLifecyclePauseTests(unittest.TestCase):
    def test_resume_keeps_existing_runtime(self) -> None:
        root = MagicMock()
        root.after = MagicMock()
        bot = MagicMock()
        lifecycle = BotLifecycleManager(
            root=root,
            config=MagicMock(window_id=123),
            mob_catalog=[],
            session=MagicMock(),
            viiper=MagicMock(),
        )
        lifecycle._bot = bot
        lifecycle._state = BotState.PAUSED

        lifecycle.resume()

        bot.resume.assert_called_once()
        self.assertIs(lifecycle._bot, bot)
        self.assertEqual(lifecycle.state, BotState.RUNNING)

    def test_start_while_paused_is_ignored(self) -> None:
        root = MagicMock()
        lifecycle = BotLifecycleManager(
            root=root,
            config=MagicMock(),
            mob_catalog=[MagicMock()],
            session=MagicMock(),
            viiper=MagicMock(),
        )
        lifecycle._state = BotState.PAUSED
        lifecycle._bot = MagicMock()

        lifecycle.start(config_snapshot=MagicMock(), session_id="test")

        lifecycle._bot.stop.assert_not_called()
        self.assertEqual(lifecycle.state, BotState.PAUSED)


if __name__ == "__main__":
    unittest.main()
