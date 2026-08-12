"""Explicitly configured mob behavior tests."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pybot.runtime.constants import CELL_SIZE_PX
from pybot.runtime.mob_behaviors import (
    ConfiguredMobBehavior,
    MobBehavior,
    get_configured_mob_behavior,
)


class MobBehaviorTests(unittest.TestCase):
    def test_default_behavior_is_no_op(self) -> None:
        behavior = get_configured_mob_behavior(
            SimpleNamespace(
                kiting_tick_ms=0,
                kite_distance_px=None,
                debuff_scan_code=0,
                debuff_button="",
                heal_scan_code=0,
                heal_button="",
                buffs=(),
            )
        )
        self.assertIsInstance(behavior, ConfiguredMobBehavior)
        self.assertIsInstance(behavior, MobBehavior)

        backend = MagicMock()
        self.assertFalse(
            behavior.kite_after_attack(
                100, 100, backend, all_mobs=[(120, 100)]
            )
        )
        backend.move_and_double_click.assert_not_called()

    def test_unset_distance_disables_kiting_even_with_interval(self) -> None:
        behavior = ConfiguredMobBehavior(
            SimpleNamespace(
                kiting_tick_ms=1_000,
                kite_distance_px=None,
                debuff_scan_code=0,
                debuff_button="",
                heal_scan_code=0,
                heal_button="",
                buffs=(),
            )
        )
        backend = MagicMock()

        with patch("pybot.runtime.mob_behaviors.monotonic_ms", return_value=10):
            self.assertFalse(
                behavior.kite_after_attack(
                    100, 100, backend, all_mobs=[(120, 100)]
                )
            )
        backend.move_and_double_click.assert_not_called()

    def test_configured_distance_moves_away_from_mob(self) -> None:
        behavior = ConfiguredMobBehavior(
            SimpleNamespace(
                kiting_tick_ms=1,
                kite_distance_px=5 * CELL_SIZE_PX,
                debuff_scan_code=0,
                debuff_button="",
                heal_scan_code=0,
                heal_button="",
                buffs=(),
            )
        )
        backend = MagicMock()
        backend.move_and_double_click.return_value = True

        with patch("pybot.runtime.mob_behaviors.monotonic_ms", return_value=10):
            self.assertTrue(
                behavior.kite_after_attack(
                    100, 100, backend, all_mobs=[(120, 100)]
                )
            )

        backend.move_and_double_click.assert_called_once_with(
            100 - 5 * CELL_SIZE_PX, 100
        )

    def test_kiting_uses_atomic_double_click(self) -> None:
        class DoubleClickBackend:
            def __init__(self) -> None:
                self.calls: list[tuple[int, int]] = []

            def move_and_double_click(self, x: int, y: int) -> bool:
                self.calls.append((x, y))
                return True

        backend = DoubleClickBackend()
        self.assertTrue(
            get_configured_mob_behavior(
                SimpleNamespace(
                    kiting_tick_ms=1,
                    kite_distance_px=320,
                    debuff_scan_code=0,
                    debuff_button="",
                    heal_scan_code=0,
                    heal_button="",
                    buffs=(),
                )
            ).kite_after_attack(
                100, 100, backend, all_mobs=[(200, 100)]
            )
        )
        self.assertEqual(backend.calls, [(-220, 100)])

    def test_kiting_chooses_open_direction_when_centered(self) -> None:
        backend = MagicMock()
        backend.move_and_double_click.return_value = True
        behavior = ConfiguredMobBehavior(
            SimpleNamespace(
                kiting_tick_ms=1,
                kite_distance_px=320,
                debuff_scan_code=0,
                debuff_button="",
                heal_scan_code=0,
                heal_button="",
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

        backend.move_and_double_click.assert_called_once_with(100, -220)

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
        backend = MagicMock()
        backend.move_and_double_click.return_value = True
        backend.skill_click_at.return_value = True
        behavior = ConfiguredMobBehavior(settings)

        with patch("pybot.runtime.mob_behaviors.monotonic_ms", return_value=10):
            behavior.kite_after_attack(
                100, 100, backend, all_mobs=[(120, 100)]
            )
            behavior.before_attack(
                100, 100, backend, all_mobs=[(120, 100)]
            )

        backend.skill_click_at.assert_not_called()


if __name__ == "__main__":
    unittest.main()
