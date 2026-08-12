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
from pybot.runtime.constants import CELL_SIZE_PX
from pybot.runtime.mob_behaviors import (
    ConfiguredMobBehavior,
    get_configured_mob_behavior,
    kite_away_from_mobs,
)
from pybot.runtime.workers.self_buff_worker import SelfBuffWorker


class CustomMobBehaviorConfigTests(unittest.TestCase):
    def test_runtime_disables_kiting_without_distance(self) -> None:
        settings = AppSettings(
            selected_monster=1,
            mob_custom_settings={"horn": MobCustomSettings(kiting_tick_s=1.0)},
        )
        with patch(
            "pybot.config.runtime.resolve_mob_descriptor_name",
            return_value="horn",
        ):
            config = hunt_runtime_config_from_settings(settings)
        self.assertEqual(config.custom_behavior.kiting_tick_ms, 1000)
        self.assertIsNone(config.custom_behavior.kite_distance_px)

    def test_runtime_converts_selected_mob_custom_settings(self) -> None:
        settings = AppSettings(
            selected_monster=1,
            mob_custom_settings={
                "horn": MobCustomSettings(
                    kiting_tick_s=0.75,
                    kite_distance_cells=7,
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

        self.assertEqual(config.custom_behavior.kiting_tick_ms, 750)
        self.assertEqual(config.custom_behavior.kite_distance_px, 7 * CELL_SIZE_PX)
        self.assertEqual(config.custom_behavior.debuff_scan_code, 19)
        self.assertEqual(config.custom_behavior.heal_scan_code, 16)
        self.assertEqual(len(config.custom_behavior.buffs), 1)
        self.assertEqual(config.custom_behavior.buffs[0].scan_code, 59)
        self.assertEqual(config.custom_behavior.buffs[0].delay_ms, 12_000)



class ConfiguredMobBehaviorTests(unittest.TestCase):
    def test_kiting_does_not_cast_heal(self) -> None:
        settings = SimpleNamespace(
            kiting_tick_ms=1,
            kite_distance_px=320,
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
        backend.move_and_double_click.return_value = True
        backend.skill_click_at.return_value = True
        behavior = ConfiguredMobBehavior(settings)

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
            [unittest.mock.call.move_and_double_click(-220, 100)],
        )
        danger.is_safe_for_heal.assert_not_called()

    def test_kiting_uses_atomic_double_click_when_backend_supports_it(self) -> None:
        class DoubleClickBackend:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int, int]] = []

            def move_and_double_click(self, x: int, y: int) -> bool:
                self.calls.append(("double", x, y))
                return True

        backend = DoubleClickBackend()
        self.assertTrue(
            kite_away_from_mobs(
                100, 100, backend, all_mobs=[(200, 100)], distance_px=320
            )
        )
        self.assertEqual(backend.calls, [("double", -220, 100)])

    def test_kiting_click_is_fixed_distance_from_character(self) -> None:
        settings = SimpleNamespace(
            kiting_tick_ms=1,
            kite_distance_px=320,
            debuff_scan_code=0,
            heal_scan_code=0,
            buffs=(),
        )
        backend = MagicMock()
        backend.move_and_double_click.return_value = True
        behavior = ConfiguredMobBehavior(settings)

        with patch("pybot.runtime.mob_behaviors.monotonic_ms", return_value=10):
            self.assertTrue(
                behavior.kite_after_attack(
                    100,
                    100,
                    backend,
                    all_mobs=[(100, 200)],
                )
            )

        target_x, target_y = backend.move_and_double_click.call_args.args
        self.assertEqual(target_x, 100)
        # Default kite distance is 5 cells × CELL_SIZE_PX.
        self.assertEqual(abs(target_y - 100), 5 * CELL_SIZE_PX)

    def test_kiting_uses_configured_distance(self) -> None:
        settings = SimpleNamespace(
            kiting_tick_ms=1,
            # An explicit configured distance distinct from the 5-cell default.
            kite_distance_px=320,
            debuff_scan_code=0,
            heal_scan_code=0,
            buffs=(),
        )
        backend = MagicMock()
        backend.move_and_double_click.return_value = True
        behavior = ConfiguredMobBehavior(settings)

        with patch("pybot.runtime.mob_behaviors.monotonic_ms", return_value=10):
            self.assertTrue(
                behavior.kite_after_attack(100, 100, backend, all_mobs=[(200, 100)])
            )

        self.assertEqual(backend.move_and_double_click.call_args.args, (-220, 100))

    def test_kiting_chooses_open_direction_when_mobs_are_centered(self) -> None:
        backend = MagicMock()
        backend.move_and_double_click.return_value = True
        behavior = ConfiguredMobBehavior(
            SimpleNamespace(
                kiting_tick_ms=1,
                kite_distance_px=320,
                debuff_scan_code=0,
                heal_scan_code=0,
                buffs=(),
            )
        )

        with patch("pybot.runtime.mob_behaviors.monotonic_ms", return_value=10):
            self.assertTrue(
                behavior.kite_after_attack(
                    100,
                    100,
                    backend,
                    all_mobs=[(100, 100), (100, 100)],
                )
            )

        # Fully centered is symmetric; deterministic compass order chooses up.
        self.assertEqual(backend.move_and_double_click.call_args.args, (100, -220))

    def test_kiting_centered_mobs_choose_less_occupied_direction(self) -> None:
        backend = MagicMock()
        backend.move_and_double_click.return_value = True
        behavior = ConfiguredMobBehavior(
            SimpleNamespace(
                kiting_tick_ms=1,
                kite_distance_px=320,
                debuff_scan_code=0,
                heal_scan_code=0,
                buffs=(),
            )
        )

        with patch("pybot.runtime.mob_behaviors.monotonic_ms", return_value=10):
            self.assertTrue(
                behavior.kite_after_attack(
                    100,
                    100,
                    backend,
                    # The average is centered, but all tracked mobs are on
                    # the horizontal axis, so up is the least occupied sector.
                    all_mobs=[(120, 100), (120, 100), (60, 100)],
                )
            )

        self.assertEqual(backend.move_and_double_click.call_args.args, (100, -220))

    def test_casts_debuff_once_per_target_and_retries_failed_cast(self) -> None:
        settings = SimpleNamespace(
            debuff_button="r",
            debuff_scan_code=19,
            heal_scan_code=0,
            kiting_tick_ms=0,
            buffs=(),
        )
        behavior = ConfiguredMobBehavior(settings)
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
            [
                unittest.mock.call(19, 100, 200, move_delay_s=0.0),
                unittest.mock.call(19, 100, 200, move_delay_s=0.0),
            ],
        )
        marked.assert_called_once()

    def test_skips_debuff_for_already_prepared_target(self) -> None:
        settings = SimpleNamespace(debuff_button="r", debuff_scan_code=19)
        behavior = ConfiguredMobBehavior(settings)
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

    def test_empty_kiting_interval_disables_kiting(self) -> None:
        settings = SimpleNamespace(
            kiting_tick_ms=0,
            kite_distance_px=320,
            debuff_button="",
            debuff_scan_code=0,
            heal_button="",
            heal_scan_code=0,
            buffs=(),
        )
        backend = MagicMock()
        backend.move_and_double_click.return_value = True
        behavior = get_configured_mob_behavior(settings)

        behavior.before_attack(100, 100, backend, all_mobs=[(120, 100)])
        behavior.kite_after_attack(100, 100, backend, all_mobs=[(120, 100)])

        backend.move_and_double_click.assert_not_called()


class SelfBuffWorkerTests(unittest.TestCase):
    def test_pre_clear_window_paces_poll_instead_of_spinning(self) -> None:
        """A recovered-hunt pre-clear window must poll, not spin on the GIL.

        After a danger escape (``trusted_clear=False``) combat is admitted
        (``wait_while_combat_blocked`` returns immediately) while startup
        actions stay blocked until the first discovery scan confirms the
        landing area. The old loop continued without sleeping in that state,
        spinning hot and starving the very scan that would release it — the
        observed 8-13 s post-teleport freeze.
        """
        stop = threading.Event()
        waits: list[float] = []

        class StopEvent:
            def is_set(self) -> bool:
                return stop.is_set()

            def wait(self, timeout: float) -> bool:
                waits.append(timeout)
                if len(waits) >= 3:
                    stop.set()
                return stop.is_set()

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
            hunt_generation=1,
            # Recovered hunt: area not yet confirmed clear by a scan.
            should_run_startup_actions=lambda: False,
            # Combat is admitted in the pre-clear window: returns instantly.
            wait_while_combat_blocked=lambda _timeout: True,
            character_screen_pos=lambda: (300, 350),
            character_action_gate=CharacterActionGate(),
        )

        worker = SelfBuffWorker(
            ctx, SimpleNamespace(skill_click_at=MagicMock(return_value=True))
        )
        result = worker.process_pending()

        # The sequence bails on stop; every blocked iteration must have
        # slept a real poll so the loop cannot hold the GIL while waiting
        # for the scan to clear the area.
        self.assertFalse(result)
        self.assertGreaterEqual(len(waits), 3)
        self.assertTrue(all(timeout > 0 for timeout in waits))
