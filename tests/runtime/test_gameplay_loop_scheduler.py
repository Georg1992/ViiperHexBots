"""GameplayLoop scheduling regressions."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.workers.attack_loop import GameplayLoop


class _StartupBuff:
    scan_code = 59
    delay_ms = 60_000
    button = "f1"


class _StartupTimer:
    scan_code = 60
    interval_ms = 60_000
    button = "f2"


class GameplayLoopSchedulerTests(unittest.TestCase):
    def test_startup_actions_are_not_replayed_before_all_milestones(self) -> None:
        """Periodic scheduling waits for startup buff and timer completion."""
        config = SimpleNamespace(
            custom_behavior=SimpleNamespace(buffs=(_StartupBuff(),)),
            skill_timers=(_StartupTimer(),),
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
        ctx.begin_hunt_startup(require_buffs=True, require_timers=True)

        periodic_states: list[tuple[bool, bool]] = []

        class Buffs:
            def __init__(self) -> None:
                self.startup_calls = 0
                self.periodic_calls = 0

            def process_pending(self, *, startup_only: bool = False) -> bool:
                self.assertTrue(startup_only)
                self.startup_calls += 1
                ctx.mark_startup_buffs_done()
                return True

            def execute_buff(self, _scan_code: int) -> bool:
                self.periodic_calls += 1
                periodic_states.append(
                    (
                        ctx.startup_buffs_done.is_set(),
                        ctx.startup_timers_done.is_set(),
                    )
                )
                return True

            def last_success_ms(self, _scan_code: int) -> int:
                return 100

            @staticmethod
            def assertTrue(value: bool) -> None:
                if not value:
                    raise AssertionError("periodic callback used during startup")

        class Timers:
            def __init__(self) -> None:
                self.startup_calls = 0
                self.periodic_calls = 0

            def process_pending(self, *, startup_only: bool = False) -> bool:
                self.assertTrue(startup_only)
                self.startup_calls += 1
                # Leave the timer milestone incomplete for one loop iteration.
                if self.startup_calls >= 2:
                    ctx.mark_startup_timers_done()
                return True

            def execute_timer(self, _scan_code: int) -> bool:
                self.periodic_calls += 1
                periodic_states.append(
                    (
                        ctx.startup_buffs_done.is_set(),
                        ctx.startup_timers_done.is_set(),
                    )
                )
                return True

            def last_success_ms(self, _scan_code: int) -> int:
                return 200

            @staticmethod
            def assertTrue(value: bool) -> None:
                if not value:
                    raise AssertionError("periodic callback used during startup")

        buffs = Buffs()
        timers = Timers()

        class Attack:
            def process_pending(self) -> bool:
                ctx.stop_event.set()
                return False

        gameplay = GameplayLoop(
            ctx,
            attack=Attack(),
            buffs=buffs,
            timers=timers,
        )
        gameplay.run()

        self.assertEqual(buffs.startup_calls, 2)
        self.assertEqual(timers.startup_calls, 2)
        self.assertTrue(periodic_states)
        self.assertTrue(all(buffs_done and timers_done for buffs_done, timers_done in periodic_states))


if __name__ == "__main__":
    unittest.main()
