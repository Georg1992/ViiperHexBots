"""Critical danger is signalled by observation and consumed by GameplayLoop."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from pybot.game_state import PlayerVitals
from pybot.runtime.danger_detector import DangerDetector, DangerLevel
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.workers.attack_loop import GameplayLoop
from pybot.runtime.workers.self_buff_worker import SelfBuffWorker


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
        ctx.danger_detector = SimpleNamespace(
            danger_level=lambda: DangerLevel.CRITICAL,
        )
        return ctx

    def test_gameplay_owner_consumes_critical_escape(self) -> None:
        """The registered gameplay owner performs the urgent teleport itself."""
        ctx = self._context()
        ctx.request_critical_danger()
        teleport = MagicMock()
        teleport.danger_teleport.return_value = True
        input_backend = MagicMock()
        input_backend.begin_session.return_value = True

        gameplay = GameplayLoop(
            ctx,
            attack=MagicMock(),
            teleport=teleport,
            input_backend=input_backend,
        )

        self.assertTrue(gameplay._process_critical_danger())
        self.assertFalse(ctx.critical_danger_requested.is_set())
        teleport.danger_teleport.assert_called_once_with(reason="critical_hunt")
        input_backend.begin_session.assert_called_once_with()

    def test_hp_observer_cancels_input_but_does_not_teleport(self) -> None:
        """The metric producer only publishes urgency and requests cancellation."""
        ctx = self._context()
        vitals = PlayerVitals()
        cancel = MagicMock()
        ctx.cancel_gameplay_input = cancel
        detector = DangerDetector(ctx, vitals=vitals)

        vitals.publish_hp(100, 100)
        detector._poll_hp()
        vitals.publish_hp(40, 100)
        detector._poll_hp()

        cancel.assert_called_once_with()
        self.assertTrue(ctx.critical_danger_requested.is_set())

    def test_preempted_session_timeout_releases_temporary_gates(self) -> None:
        """A failed storage/heal handoff cannot leave gameplay permanently gated."""
        ctx = self._context()
        ctx.request_critical_danger()
        ctx.storage_event.set()
        # The real GateController wait path times out while storage remains
        # held; no mock should hide the ownership cleanup being tested.
        gameplay = GameplayLoop(
            ctx,
            attack=MagicMock(),
            teleport=MagicMock(),
            input_backend=MagicMock(),
        )

        self.assertFalse(gameplay._process_critical_danger())
        self.assertTrue(ctx.critical_danger_requested.is_set())
        self.assertFalse(ctx.danger_escape_active.is_set())
        self.assertFalse(ctx.critical_danger_escape_active.is_set())
        self.assertFalse(ctx.sitting_event.is_set())

    def test_startup_wait_aborts_when_critical_request_arrives(self) -> None:
        """Startup postponement must yield to the gameplay owner's urgent step."""
        ctx = self._context()
        ctx.config.custom_behavior.buffs = (
            SimpleNamespace(scan_code=59, delay_ms=10_000, button="f1"),
        )
        ctx.wait_while_combat_blocked = MagicMock(
            side_effect=lambda _timeout: (
                ctx.request_critical_danger(), False
            )[1]
        )
        worker = SelfBuffWorker(ctx, MagicMock())

        self.assertFalse(
            worker._run_startup_sequence(
                tuple(ctx.config.custom_behavior.buffs),
                expected_generation=ctx.hunt_generation,
            )
        )
        self.assertTrue(ctx.critical_danger_requested.is_set())

        # The same owner that aborted startup must consume the request and
        # perform the escape; no second controller is involved.
        teleport = MagicMock()
        teleport.danger_teleport.return_value = True
        gameplay = GameplayLoop(
            ctx,
            attack=MagicMock(),
            teleport=teleport,
            input_backend=MagicMock(),
        )
        self.assertTrue(gameplay._process_critical_danger())
        teleport.danger_teleport.assert_called_once_with(reason="critical_hunt")


if __name__ == "__main__":
    unittest.main()
