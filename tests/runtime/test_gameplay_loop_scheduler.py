"""GameplayLoop scheduling regressions."""

from __future__ import annotations

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


class GameplayLoopRecoveredHuntTests(unittest.TestCase):
    def test_attack_keeps_running_in_pre_clear_window_after_recovery_landing(self) -> None:
        """A populated landing after a recovered hunt must stay attackable.

        ``begin_new_hunt(trusted_clear=False)`` (a recovered sit landing or a
        trusted start downgraded by a populated first scan) clears the startup
        milestones and keeps combat live before the first empty discovery
        scan. The milestone gate must not freeze attack while the landing
        area still contains mobs.
        """
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
        # Simulate the danger-escape fresh transition: milestones pending,
        # area not yet confirmed clear, combat admitted for the clear pass.
        ctx.gates.startup.begin_new_hunt(trusted_clear=False)
        self.assertTrue(ctx.gates.startup.is_combat_ready())

        attack_calls: list[int] = []

        class Buffs:
            def process_pending(self, *, startup_only: bool = False) -> bool:
                # Startup buffs wait for the first empty scan; never complete
                # here so the milestones stay pending throughout the test.
                return False

            def execute_buff(self, _scan_code: int) -> bool:
                return False

            def last_success_ms(self, _scan_code: int):
                return None

        class Timers:
            def process_pending(self, *, startup_only: bool = False) -> bool:
                return False

            def execute_timer(self, _scan_code: int) -> bool:
                return False

            def last_success_ms(self, _scan_code: int):
                return None

        class Attack:
            def process_pending(self) -> bool:
                attack_calls.append(len(attack_calls))
                if len(attack_calls) >= 2:
                    ctx.stop_event.set()
                return True

        gameplay = GameplayLoop(
            ctx,
            attack=Attack(),
            buffs=Buffs(),
            timers=Timers(),
        )
        gameplay.run()

        # Attack must have run while startup milestones were still pending,
        # proving the pre-clear window keeps combat live.
        self.assertGreaterEqual(len(attack_calls), 2)
        self.assertFalse(ctx.gates.startup.area_clear.is_set())

    def test_fresh_start_keeps_attack_gated_until_milestones(self) -> None:
        """A trusted fresh start does not attack while milestones are pending."""
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
        # Trusted fresh start: area pre-confirmed clear, milestones pending.
        self.assertTrue(ctx.gates.startup.area_clear.is_set())
        self.assertFalse(ctx.gates.startup.is_combat_ready())

        attack_calls: list[int] = []
        buff_ticks = {"n": 0}

        class Buffs:
            def process_pending(self, *, startup_only: bool = False) -> bool:
                # Milestones stay pending; end the loop after a few ticks so
                # the test proves attack was never admitted during startup.
                buff_ticks["n"] += 1
                if buff_ticks["n"] >= 3:
                    ctx.stop_event.set()
                return False

            def execute_buff(self, _scan_code: int) -> bool:
                return False

            def last_success_ms(self, _scan_code: int):
                return None

        class Timers:
            def process_pending(self, *, startup_only: bool = False) -> bool:
                return False

            def execute_timer(self, _scan_code: int) -> bool:
                return False

            def last_success_ms(self, _scan_code: int):
                return None

        class Attack:
            def process_pending(self) -> bool:
                attack_calls.append(len(attack_calls))
                return True

        gameplay = GameplayLoop(
            ctx,
            attack=Attack(),
            buffs=Buffs(),
            timers=Timers(),
        )
        gameplay.run()

        # Fresh trusted starts gate attack on both startup milestones: with
        # the area already clear and milestones pending, combat must stay
        # closed (unlike a recovered hunt's pre-clear window).
        self.assertEqual(attack_calls, [])


if __name__ == "__main__":
    unittest.main()
