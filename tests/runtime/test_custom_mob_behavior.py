"""Per-mob custom behavior runtime tests."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pybot.config.runtime import hunt_runtime_config_from_settings
from pybot.config.schema import AppSettings, MobCustomSettings
from pybot.game_state import PlayerVitals
from pybot.config.runtime import SelfBuffRuntime, CustomBehaviorRuntime
from pybot.runtime.gate_controller import CharacterActionGate
from pybot.runtime.mob_behaviors import (
    AnubisBehavior,
    ConfiguredMobBehavior,
    get_configured_mob_behavior,
)
from pybot.runtime.workers.self_buff_worker import SelfBuffWorker


class CustomMobBehaviorConfigTests(unittest.TestCase):
    def test_runtime_converts_selected_mob_custom_settings(self) -> None:
        settings = AppSettings(
            selected_monster=1,
            mob_custom_settings={
                "horn": MobCustomSettings(
                    kiting_tick_s=0.75,
                    debuff_button="r",
                    heal_button="q",
                    buff1_button="f1",
                    buff1_delay_s=12,
                    buff2_button="f2",
                    buff2_delay_s=0,
                )
            },
        )

        with patch(
            "pybot.config.runtime.resolve_mob_descriptor_name",
            return_value="horn",
        ), patch(
            "pybot.config.runtime.key_name_to_scan_code",
            side_effect=lambda key: {"q": 16, "r": 19, "f1": 59, "f2": 60}.get(
                key.lower(), 0
            ),
        ):
            config = hunt_runtime_config_from_settings(settings)

        self.assertTrue(config.custom_behavior.configured)
        self.assertEqual(config.custom_behavior.kiting_tick_ms, 750)
        self.assertEqual(config.custom_behavior.debuff_scan_code, 19)
        self.assertEqual(config.custom_behavior.heal_scan_code, 16)
        self.assertEqual(len(config.custom_behavior.buffs), 1)
        self.assertEqual(config.custom_behavior.buffs[0].scan_code, 59)
        self.assertEqual(config.custom_behavior.buffs[0].delay_ms, 12_000)



class ConfiguredMobBehaviorTests(unittest.TestCase):
    def test_kiting_does_not_cast_heal(self) -> None:
        settings = SimpleNamespace(
            configured=True,
            kiting_tick_ms=1,
            debuff_button="",
            debuff_scan_code=0,
            heal_button="q",
            heal_scan_code=16,
            buffs=(),
        )
        vitals = PlayerVitals()
        vitals.publish_hp(80, 100)
        danger = MagicMock()
        backend = MagicMock()
        backend.move_and_click.return_value = True
        backend.skill_click_at.return_value = True
        behavior = ConfiguredMobBehavior(settings, vitals, danger)

        with patch("pybot.runtime.mob_behaviors.monotonic_ms", return_value=10):
            behavior.kite_after_attack(
                100,
                100,
                backend,
                all_mobs=[(120, 100)],
            )
            behavior.before_attack(
                100,
                100,
                backend,
                all_mobs=[(120, 100)],
            )

        self.assertEqual(
            backend.method_calls,
            [unittest.mock.call.move_and_click(80, 100)],
        )
        danger.is_safe_for_heal.assert_not_called()

    def test_casts_debuff_once_per_target_and_retries_failed_cast(self) -> None:
        settings = SimpleNamespace(
            debuff_scan_code=19,
            heal_scan_code=0,
            kiting_tick_ms=0,
            buffs=(),
        )
        behavior = ConfiguredMobBehavior(settings, PlayerVitals(), MagicMock())
        backend = MagicMock()
        backend.skill_click_at.side_effect = [False, True]
        marked = MagicMock(return_value=True)

        self.assertFalse(
            behavior.prepare_target(
                7, 100, 200, backend,
                target_debuffed=False,
                mark_debuffed=marked,
            )
        )
        self.assertTrue(
            behavior.prepare_target(
                7, 100, 200, backend,
                target_debuffed=False,
                mark_debuffed=marked,
            )
        )
        self.assertEqual(
            backend.skill_click_at.call_args_list,
            [unittest.mock.call(19, 100, 200), unittest.mock.call(19, 100, 200)],
        )
        marked.assert_called_once()

    def test_skips_debuff_for_already_prepared_target(self) -> None:
        settings = SimpleNamespace(debuff_scan_code=19)
        behavior = ConfiguredMobBehavior(settings, PlayerVitals(), MagicMock())
        backend = MagicMock()
        marker = MagicMock()

        self.assertTrue(
            behavior.prepare_target(
                7, 100, 200, backend,
                target_debuffed=True,
                mark_debuffed=marker,
            )
        )
        backend.skill_click_at.assert_not_called()
        marker.assert_not_called()

    def test_saved_empty_anubis_settings_keep_legacy_kiting_in_cycle(self) -> None:
        settings = SimpleNamespace(
            configured=True,
            kiting_tick_ms=0,
            debuff_button="",
            debuff_scan_code=0,
            heal_button="",
            heal_scan_code=0,
            buffs=(),
        )
        vitals = PlayerVitals()
        danger = MagicMock()
        legacy = AnubisBehavior()
        backend = MagicMock()
        backend.move_and_click.return_value = True
        behavior = get_configured_mob_behavior(
            settings, vitals, danger, legacy_behavior=legacy
        )

        behavior.before_attack(100, 100, backend, all_mobs=[(120, 100)])
        behavior.kite_after_attack(100, 100, backend, all_mobs=[(120, 100)])

        backend.move_and_click.assert_called_once_with(80, 100)


class SelfBuffWorkerTests(unittest.TestCase):
    def test_casts_configured_buffs_in_order_with_one_second_startup_gap(self) -> None:
        stop = threading.Event()
        clock = {"ms": 0}
        safe = {"value": False}
        casts: list[tuple[int, int, int]] = []

        class StopEvent:
            def is_set(self) -> bool:
                return stop.is_set()

            def wait(self, timeout: float) -> bool:
                clock["ms"] += int(timeout * 1000)
                if not safe["value"]:
                    safe["value"] = True
                return stop.is_set()

        def cast(scan_code: int, x: int, y: int) -> bool:
            casts.append((scan_code, x, y))
            if len(casts) == 2:
                stop.set()
            return True

        ctx = SimpleNamespace(
            config=SimpleNamespace(
                skill_timers=(),
                custom_behavior=CustomBehaviorRuntime(
                    buffs=(
                        SelfBuffRuntime("f1", 59, 1000),
                        SelfBuffRuntime("f2", 60, 2000),
                    )
                )
            ),
            logger=SimpleNamespace(behavior=MagicMock()),
            stop_event=StopEvent(),
            is_stopped=stop.is_set,
            should_run_combat=lambda: safe["value"] and not stop.is_set(),
            wait_while_combat_blocked=lambda _timeout: safe.__setitem__("value", True) or True,
            character_screen_pos=lambda: (300, 350),
            character_action_gate=CharacterActionGate(),
        )

        with patch(
            "pybot.runtime.workers.self_buff_worker.monotonic_ms",
            side_effect=lambda: clock["ms"],
        ):
            SelfBuffWorker(ctx, SimpleNamespace(skill_click_at=cast)).run()

        self.assertEqual(casts, [(59, 300, 350), (60, 300, 350)])
        self.assertEqual(clock["ms"], 1000)

    def test_stops_startup_sequence_while_waiting_for_normal_timers(self) -> None:
        stop = threading.Event()

        class StopEvent:
            def is_set(self) -> bool:
                return stop.is_set()

            def wait(self, _timeout: float) -> bool:
                stop.set()
                return True

        class StartupEvent:
            def wait(self, _timeout: float) -> bool:
                stop.set()
                return False

        ctx = SimpleNamespace(
            config=SimpleNamespace(
                skill_timers=(SimpleNamespace(scan_code=59, interval_ms=1000),),
                custom_behavior=CustomBehaviorRuntime(
                    buffs=(SelfBuffRuntime("f1", 59, 1000),)
                )
            ),
            startup_timers_done=StartupEvent(),
            logger=SimpleNamespace(behavior=MagicMock()),
            stop_event=StopEvent(),
            is_stopped=stop.is_set,
            should_run_combat=lambda: not stop.is_set(),
            should_run_startup_actions=lambda: False,
            wait_while_combat_blocked=lambda _timeout: False,
            character_screen_pos=lambda: (300, 350),
            character_action_gate=CharacterActionGate(),
        )
        backend = SimpleNamespace(skill_click_at=MagicMock(return_value=True))

        SelfBuffWorker(ctx, backend).run()

        backend.skill_click_at.assert_not_called()

    def test_casts_buff_on_character_immediately_after_timer_startup(self) -> None:
        stop = threading.Event()
        clock = {"ms": 0}
        casts: list[tuple[int, int, int]] = []

        class StopEvent:
            def is_set(self) -> bool:
                return stop.is_set()

            def wait(self, timeout: float) -> bool:
                clock["ms"] += int(timeout * 1000)
                return stop.is_set()

            def set(self) -> None:
                stop.set()

        def cast(scan_code: int, x: int, y: int) -> bool:
            casts.append((scan_code, x, y))
            stop.set()
            return True

        ctx = SimpleNamespace(
            config=SimpleNamespace(
                custom_behavior=CustomBehaviorRuntime(
                    buffs=(SelfBuffRuntime("f1", 59, 1000),)
                )
            ),
            logger=SimpleNamespace(behavior=MagicMock()),
            stop_event=StopEvent(),
            is_stopped=stop.is_set,
            should_run_combat=lambda: not stop.is_set(),
            wait_while_combat_blocked=lambda _timeout: not stop.is_set(),
            character_screen_pos=lambda: (300, 350),
            character_action_gate=CharacterActionGate(),
        )

        with patch(
            "pybot.runtime.workers.self_buff_worker.monotonic_ms",
            side_effect=lambda: clock["ms"],
        ):
            SelfBuffWorker(ctx, SimpleNamespace(skill_click_at=cast)).run()

        self.assertEqual(casts, [(59, 300, 350)])
        self.assertEqual(clock["ms"], 0)

    def test_periodic_buff_casts_under_shared_burst_flag(self) -> None:
        """A periodic buff cast claims buff priority on the shared gate."""
        stop = threading.Event()
        clock = {"ms": 0}
        safe = {"value": True}
        gate = CharacterActionGate()
        casts: list[tuple[int, int, int]] = []
        burst_at_cast: list[bool] = []

        class StopEvent:
            def is_set(self) -> bool:
                return stop.is_set()

            def wait(self, timeout: float) -> bool:
                clock["ms"] += int(timeout * 1000)
                if len(casts) >= 2:
                    stop.set()
                return stop.is_set()

        def cast(scan_code: int, x: int, y: int) -> bool:
            burst_at_cast.append(gate.buff_burst_active())
            casts.append((scan_code, x, y))
            return True

        ctx = SimpleNamespace(
            config=SimpleNamespace(
                skill_timers=(),
                custom_behavior=CustomBehaviorRuntime(
                    buffs=(SelfBuffRuntime("f1", 59, 1000),)
                )
            ),
            logger=SimpleNamespace(behavior=MagicMock()),
            stop_event=StopEvent(),
            is_stopped=stop.is_set,
            should_run_combat=lambda: safe["value"] and not stop.is_set(),
            wait_while_combat_blocked=lambda _timeout: safe["value"],
            character_screen_pos=lambda: (300, 350),
            character_action_gate=gate,
        )

        with patch(
            "pybot.runtime.workers.self_buff_worker.monotonic_ms",
            side_effect=lambda: clock["ms"],
        ):
            SelfBuffWorker(ctx, SimpleNamespace(skill_click_at=cast)).run()

        # Startup cast (no burst) + one periodic re-cast (burst held).
        self.assertEqual(len(casts), 2)
        self.assertFalse(burst_at_cast[0])
        self.assertTrue(burst_at_cast[1])
        # The burst is released once the due-buff loop finishes.
        self.assertFalse(gate.buff_burst_active())


if __name__ == "__main__":
    unittest.main()
