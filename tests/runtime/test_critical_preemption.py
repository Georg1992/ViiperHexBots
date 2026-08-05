"""Critical danger must preempt a blocked gameplay action."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from pybot.runtime.danger_detector import DangerLevel
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.workers.attack_loop import GameplayLoop
from pybot.runtime.workers.critical_danger_worker import CriticalDangerWorker


class _BlockingStartup:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def process_pending(self, *, startup_only: bool = False) -> bool:
        self.assert_startup_only(startup_only)
        self.entered.set()
        self.release.wait(timeout=2.0)
        return False

    @staticmethod
    def assert_startup_only(startup_only: bool) -> None:
        if not startup_only:
            raise AssertionError("test callback was used as a periodic action")


class CriticalPreemptionTests(unittest.TestCase):
    def _context(self) -> HuntRuntimeContext:
        config = SimpleNamespace(
            custom_behavior=SimpleNamespace(buffs=()),
            skill_timers=(),
        )
        ctx = HuntRuntimeContext(
            config=config,
            logger=MagicMock(),
            tracks=MagicMock(),
            policy=MagicMock(),
            capture=MagicMock(),
            detector=MagicMock(),
            tracker=MagicMock(),
            validation=MagicMock(),
            control=MagicMock(),
        )
        ctx.mark_running()
        ctx.danger_detector = SimpleNamespace(
            danger_level=lambda: DangerLevel.CRITICAL,
        )
        return ctx

    def test_critical_escape_does_not_wait_for_gameplay_owner(self) -> None:
        """A blocked startup callback cannot strand critical danger forever."""
        ctx = self._context()
        startup = _BlockingStartup()
        gameplay = GameplayLoop(
            ctx,
            attack=MagicMock(),
            buffs=startup,
        )
        gameplay_thread = threading.Thread(target=gameplay.run, daemon=True)
        gameplay_thread.start()
        self.assertTrue(startup.entered.wait(timeout=1.0))

        # The normal runtime wiring adds the emergency worker separately from
        # GameplayLoop; this regression exercises the same independent ownership
        # boundary directly so the blocked callback remains untouched.
        ctx.request_critical_danger()
        teleport = MagicMock()
        teleport.danger_teleport.return_value = True
        critical = CriticalDangerWorker(ctx, teleport)

        completed = threading.Event()

        def run_escape() -> None:
            critical.process_pending()
            completed.set()

        escape_thread = threading.Thread(target=run_escape, daemon=True)
        escape_thread.start()
        self.assertTrue(completed.wait(timeout=1.0))
        self.assertFalse(ctx.critical_danger_requested.is_set())
        teleport.danger_teleport.assert_called_once_with(reason="critical_hunt")
        # The gameplay callback is still blocked; critical handling did not
        # depend on it returning to the top of its loop.
        self.assertTrue(gameplay_thread.is_alive())

        ctx.stop_event.set()
        startup.release.set()
        gameplay_thread.join(timeout=1.0)
        escape_thread.join(timeout=1.0)
        self.assertFalse(gameplay_thread.is_alive())
        self.assertFalse(escape_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
