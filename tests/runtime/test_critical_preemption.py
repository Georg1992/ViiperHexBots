"""Danger observation and the single gameplay-owned escape path."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from pybot.game_state import PlayerVitals
from pybot.runtime.danger_detector import DangerController, DangerDetector, DangerLevel
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.workers.gameplay_loop import GameplayLoop


class CriticalPreemptionTests(unittest.TestCase):
    def _context(self) -> HuntRuntimeContext:
        config = SimpleNamespace(
            custom_behavior=SimpleNamespace(buffs=()),
            skill_timers=(),
            sit_on_low_sp_scan_code=0,
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
        return ctx

    def test_gameplay_owner_consumes_critical_fact(self) -> None:
        """GameplayLoop delegates one critical escape to the controller."""
        ctx = self._context()
        ctx.danger_detector = SimpleNamespace(
            danger_level=lambda: DangerLevel.CRITICAL,
        )
        controller = MagicMock()
        controller.process.return_value = True
        ctx.danger_controller = controller
        gameplay = GameplayLoop(
            ctx,
            attack=MagicMock(),
            teleport=MagicMock(),
            input_backend=MagicMock(),
        )

        self.assertTrue(gameplay._process_critical_danger())
        controller.process.assert_called_once_with(seated=False)

    def test_hp_observer_only_records_fact_and_wakes_gameplay(self) -> None:
        """The observation thread never cancels input or performs teleport."""
        ctx = self._context()
        vitals = PlayerVitals()
        cancel = MagicMock()
        ctx.cancel_gameplay_input = cancel
        detector = DangerDetector(
            ctx,
            vitals=vitals,
            wake_event=ctx.danger_wake,
        )

        vitals.publish_hp(100, 100)
        detector._poll_hp()
        vitals.publish_hp(40, 100)
        detector._poll_hp()

        cancel.assert_not_called()
        self.assertTrue(ctx.danger_wake.is_set())
        self.assertEqual(detector.danger_level(), DangerLevel.CRITICAL)

    def test_controller_serializes_one_escape_transaction(self) -> None:
        """Controller owns the gate and delegates one complete teleport."""
        ctx = self._context()
        vitals = PlayerVitals()
        detector = DangerDetector(ctx, vitals=vitals)
        teleport = MagicMock()
        teleport.danger_teleport.return_value = True
        controller = DangerController(ctx, detector, teleport, MagicMock())

        vitals.publish_hp(100, 100)
        detector._poll_hp()
        vitals.publish_hp(40, 100)
        detector._poll_hp()

        self.assertTrue(controller.process(seated=False))
        teleport.danger_teleport.assert_called_once_with(
            reason="critical_hunt",
            prefer_safe_key=False,
        )
        self.assertFalse(ctx.danger_escape_active.is_set())

    def test_failed_escape_does_not_leave_gate_held(self) -> None:
        """A failed teleport is retryable but cannot strand the input gate."""
        ctx = self._context()
        vitals = PlayerVitals()
        detector = DangerDetector(ctx, vitals=vitals)
        teleport = MagicMock()
        teleport.danger_teleport.return_value = False
        controller = DangerController(ctx, detector, teleport, MagicMock())

        vitals.publish_hp(100, 100)
        detector._poll_hp()
        vitals.publish_hp(70, 100)
        detector._poll_hp()

        self.assertFalse(controller.process(seated=False))
        self.assertFalse(ctx.danger_escape_active.is_set())
        teleport.danger_teleport.assert_called_once()
