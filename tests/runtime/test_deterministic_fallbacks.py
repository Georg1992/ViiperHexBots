"""Deterministic dependency and configuration fallback behavior."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pybot.app.bot_controller import BotController
from pybot.config.runtime import hunt_runtime_config_from_settings
from pybot.config.schema import AppSettings
from pybot.runtime.gate_controller import GateController
from pybot.runtime.input.input_backend import perform_if_allowed
from pybot.runtime.workers.attack_loop import AttackLoop


class FalseValue:
    """A valid injected object that deliberately has false truthiness."""

    def __bool__(self) -> bool:
        return False


class DeterministicFallbackTests(unittest.TestCase):
    def test_injected_controller_dependencies_are_not_replaced_by_truthiness(self) -> None:
        overlay = FalseValue()
        vitals = FalseValue()
        controller = BotController(
            app_config=MagicMock(),
            session_id="deterministic-fallbacks",
            overlay=overlay,
            vitals=vitals,
        )

        self.assertIs(controller._overlay, overlay)
        self.assertIs(controller._vitals, vitals)

    def test_attack_loop_keeps_false_value_dependencies(self) -> None:
        behavior = FalseValue()
        vitals = FalseValue()
        loop = AttackLoop(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            mob_behavior=behavior,
            vitals=vitals,
        )

        self.assertIs(loop._mob_behavior, behavior)
        self.assertIs(loop._vitals, vitals)

    def test_gate_controller_keeps_false_value_startup(self) -> None:
        startup = FalseValue()
        gates = GateController(startup=startup)

        self.assertIs(gates.startup, startup)

    def test_magicmock_lifecycle_uses_direct_compatibility_path(self) -> None:
        action = MagicMock(return_value=True)
        lifecycle = MagicMock()

        self.assertTrue(
            perform_if_allowed(
                MagicMock(),
                lambda: True,
                action,
                lifecycle=lifecycle,
            )
        )
        action.assert_called_once_with()
        lifecycle.perform_input_if_allowed.assert_not_called()

    def test_invalid_search_range_fails_instead_of_using_default(self) -> None:
        for value in (0, 8, 17, -1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Search range"):
                    hunt_runtime_config_from_settings(
                        AppSettings(search_range=value)
                    )

    def test_valid_search_range_is_preserved(self) -> None:
        config = hunt_runtime_config_from_settings(AppSettings(search_range=9))

        self.assertEqual(config.search_range_cells, 9)

    def test_empty_hunt_mode_is_not_replaced_by_a_default(self) -> None:
        config = hunt_runtime_config_from_settings(
            AppSettings(hunt_mode="teleport"),
            hunt_mode="",
        )

        self.assertEqual(config.hunt_mode, "")


if __name__ == "__main__":
    unittest.main()
