"""Sit/stand: one press each, no pose-driven re-toggle; hunt until SP recovers."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import MagicMock

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import (
    SIT_KEY_SETTLE_S,
    SIT_LOW_SP_RATIO,
    SIT_RESUME_SP_RATIO,
)
from pybot.runtime.danger_detector import DangerDetector
from pybot.runtime.input.input_backend import ShadowInputBackend
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.workers.sit_on_low_sp_worker import SitOnLowSpWorker


class _ScriptedVitals(PlayerVitals):
    def __init__(self, ratios: list[float | None]) -> None:
        super().__init__()
        self._ratios = list(ratios)

    def sp_pair(self) -> tuple[int | None, int | None]:
        if not self._ratios:
            return 98, 100
        ratio = self._ratios.pop(0)
        if ratio is None:
            return None, None
        return int(ratio * 100), 100


class SitOnLowSpWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MagicMock()
        self.config.hwnd = 1
        self.config.sit_on_low_sp_button = "insert"
        self.config.sit_on_low_sp_scan_code = 82
        self.config.teleport_button = "q"
        self.config.teleport_scan_code = 16
        self.config.teleport_duration_ms = 10
        self.config.cell_size_px = 64
        self.config.creamy_tp_button = "w"
        self.config.creamy_tp_scan_code = 17
        self.config.take_fly_wings = False
        self.config.open_storage_steps = ()
        self.ctx = HuntRuntimeContext(
            config=self.config,
            logger=MagicMock(),
            tracks=MagicMock(),
            policy=MagicMock(),
            capture=MagicMock(),
            detector=MagicMock(),
            tracker=MagicMock(),
            validation=MagicMock(),
            control=MagicMock(),
            overlay=MagicMock(),
        )
        self.input = MagicMock(spec=ShadowInputBackend)
        from pybot.runtime.teleport import TeleportController
        self.teleport = TeleportController(self.ctx, self.input, MagicMock())
        self.teleport.teleport_until_quiet = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.danger = MagicMock(spec=DangerDetector)
        self.danger.pop_damage_detected.return_value = False
        self.danger.has_nearby_threat.return_value = False

    def _worker(self, vitals: PlayerVitals | None = None) -> SitOnLowSpWorker:
        return SitOnLowSpWorker(
            self.ctx, self.input, self.teleport,
            danger=self.danger, vitals=vitals or _ScriptedVitals([]),
        )

    def test_thresholds(self) -> None:
        self.assertAlmostEqual(SIT_LOW_SP_RATIO, 0.05)
        self.assertAlmostEqual(SIT_RESUME_SP_RATIO, 0.98)
        self.assertGreaterEqual(SIT_KEY_SETTLE_S, 0.3)

    def test_sit_presses_once_and_marks_seated(self) -> None:
        worker = self._worker()
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.assertTrue(worker.sit(82))
        self.input.toggle_key.assert_called_once_with(82)
        self.assertTrue(worker._seated)
        # Second sit is no-op — no flap.
        self.assertTrue(worker.sit(82))
        self.input.toggle_key.assert_called_once_with(82)

    def test_rejected_sit_does_not_mark_seated_or_retry_toggle(self) -> None:
        worker = self._worker()
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.input.toggle_key.return_value = False

        self.assertFalse(worker.sit(82))
        self.assertFalse(worker._seated)
        self.assertFalse(worker.sit(82))
        self.input.toggle_key.assert_has_calls([
            unittest.mock.call(82),
            unittest.mock.call(82),
        ])

    def test_interrupted_settle_after_accepted_sit_keeps_state_seated(self) -> None:
        worker = self._worker()
        self.ctx.wait_unless_stopped = lambda _t: False  # type: ignore[method-assign]
        self.input.toggle_key.return_value = True

        self.assertTrue(worker.sit(82))
        self.assertTrue(worker._seated)
        self.input.toggle_key.assert_called_once_with(82)

    def test_rejected_stand_keeps_state_for_cleanup_retry(self) -> None:
        worker = self._worker()
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._seated = True
        self.input.toggle_key.side_effect = [False, True]

        self.assertFalse(worker.stand(82))
        self.assertTrue(worker._seated)
        self.assertTrue(worker.stand(82))
        self.assertFalse(worker._seated)
        self.assertEqual(
            [call.args[0] for call in self.input.toggle_key.call_args_list],
            [82, 82],
        )

    def test_shutdown_cleanup_stands_after_stop_without_normal_toggle(self) -> None:
        worker = self._worker()
        self.ctx.stop_event.set()
        worker._seated = True
        self.input.cleanup_toggle_key.return_value = True

        self.assertTrue(worker._cleanup_stand(82))
        worker._seated = False
        self.input.cleanup_toggle_key.assert_called_once_with(82)

    def test_stand_presses_once_and_clears_seated(self) -> None:
        worker = self._worker()
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._seated = True
        self.assertTrue(worker.stand(82))
        self.input.toggle_key.assert_called_once_with(82)
        self.assertFalse(worker._seated)
        # Second stand is no-op — no flap.
        self.assertTrue(worker.stand(82))
        self.input.toggle_key.assert_called_once_with(82)

    def test_happy_path_exactly_two_taps(self) -> None:
        vitals = _ScriptedVitals(
            [SIT_LOW_SP_RATIO - 0.01, 0.50, SIT_RESUME_SP_RATIO, SIT_RESUME_SP_RATIO]
        )
        worker = self._worker(vitals)
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]

        def stop_after_two() -> None:
            while self.input.toggle_key.call_count < 2 and not self.ctx.is_stopped():
                self.ctx.stop_event.wait(0.01)
            self.ctx.stop_event.set()

        threading.Thread(target=stop_after_two, daemon=True).start()
        worker.run()
        presses = [c.args[0] for c in self.input.toggle_key.call_args_list if c.args[0] == 82]
        self.assertEqual(len(presses), 2, presses)
        self.assertFalse(worker._seated)

    def test_cleanup_failure_does_not_claim_standing(self) -> None:
        worker = self._worker()
        self.ctx.stop_event.set()
        worker._seated = True
        self.input.cleanup_toggle_key.return_value = False

        self.assertFalse(worker._cleanup_stand(82))
        self.assertTrue(worker._seated)

    def test_finally_does_not_extra_tap_after_stand(self) -> None:
        worker = self._worker(_ScriptedVitals([SIT_LOW_SP_RATIO - 0.01]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._sit_until_done = MagicMock(return_value="recovered")  # type: ignore[method-assign]
        worker._seated = False
        worker._recover_sp(SIT_LOW_SP_RATIO - 0.01)
        self.input.toggle_key.assert_not_called()
        self.assertFalse(self.ctx.sitting_event.is_set())

    def test_danger_request_survives_competing_storage_session(self) -> None:
        worker = self._worker(_ScriptedVitals([]))
        self.assertTrue(self.ctx.begin_storage_ops())
        self.ctx.request_danger_sit()

        # The sit worker cannot acquire ownership while storage is active;
        # attempting recovery must leave the queued danger request intact.
        worker._recover_sp(1.0, reason="danger", consume_danger_request=True)
        self.assertTrue(self.ctx.danger_sit_requested.is_set())

        self.ctx.end_storage_ops()
        self.assertTrue(self.ctx.pop_danger_sit_request())
        self.assertFalse(self.ctx.danger_sit_requested.is_set())

    def test_failed_sit_session_retries_with_gate_held(self) -> None:
        vitals = _ScriptedVitals([0.02] * 50)
        worker = self._worker(vitals)
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        attempts = {"n": 0}
        gate_during: list[bool] = []

        def sit_fail(_scan: int) -> str | None:
            attempts["n"] += 1
            gate_during.append(self.ctx.sitting_event.is_set())
            if attempts["n"] >= 3:
                self.ctx.stop_event.set()
            return None

        worker._sit_until_done = sit_fail  # type: ignore[method-assign]
        worker._recover_sp(0.02)
        self.assertGreaterEqual(attempts["n"], 3)
        self.assertTrue(all(gate_during))

    def test_damage_interrupt_stands_once(self) -> None:
        worker = self._worker(_ScriptedVitals([0.40] * 20))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.danger.pop_damage_detected.side_effect = [False, True]
        self.assertEqual(worker._sit_until_done(82), "interrupted")
        # enter_sit + leave_sit
        self.assertEqual(self.input.toggle_key.call_count, 2)
        self.assertFalse(worker._seated)

    def test_damage_during_area_clear_rejects_sit_spot(self) -> None:
        worker = self._worker(_ScriptedVitals([0.40] * 20))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.danger.pop_damage_detected.return_value = True
        self.teleport.danger_teleport = MagicMock(return_value=True)  # type: ignore[method-assign]

        self.assertEqual(worker._sit_until_done(82), "interrupted")
        self.input.toggle_key.assert_not_called()
        self.teleport.danger_teleport.assert_called_once_with(
            reason="sit_spot_danger"
        )
        self.assertFalse(worker._seated)

    def test_damage_while_seated_stands_before_urgent_teleport(self) -> None:
        worker = self._worker(_ScriptedVitals([0.40] * 20))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.danger.pop_damage_detected.side_effect = [False, True]
        self.teleport.danger_teleport = MagicMock(return_value=True)  # type: ignore[method-assign]
        worker._seated = False

        self.assertEqual(worker._sit_until_done(82), "interrupted")
        self.assertEqual(
            [call.args[0] for call in self.input.toggle_key.call_args_list],
            [82, 82],
        )
        self.teleport.danger_teleport.assert_called_once_with(
            reason="sit_danger"
        )
        self.assertFalse(worker._seated)

    def test_nearby_danger_while_seated_stands_before_urgent_teleport(self) -> None:
        worker = self._worker(_ScriptedVitals([0.40] * 20))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.danger.pop_damage_detected.return_value = False
        self.danger.has_nearby_threat.side_effect = [False, True]
        self.teleport.danger_teleport = MagicMock(return_value=True)  # type: ignore[method-assign]

        self.assertEqual(worker._sit_until_done(82), "interrupted")
        self.assertEqual(
            [call.args[0] for call in self.input.toggle_key.call_args_list],
            [82, 82],
        )
        self.teleport.danger_teleport.assert_called_once_with(
            reason="sit_danger"
        )
        self.assertFalse(worker._seated)

    def test_damage_recovery_finds_new_place_before_sitting_again(self) -> None:
        # The real recovery loop must see recovered SP after the interrupted
        # attempt; otherwise it correctly keeps looking for another spot.
        worker = self._worker(_ScriptedVitals([0.99, 0.99]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._sit_until_done = MagicMock(  # type: ignore[method-assign]
            side_effect=["interrupted", "recovered"]
        )
        self.teleport.teleport_to_safe_place = MagicMock(  # type: ignore[method-assign]
            side_effect=[True, True]
        )
        self.teleport.danger_teleport = MagicMock(return_value=True)  # type: ignore[method-assign]
        worker._seated = False

        worker._recover_sp(0.02)

        self.assertEqual(worker._sit_until_done.call_count, 2)
        self.assertEqual(self.teleport.teleport_to_safe_place.call_count, 2)
        self.teleport.danger_teleport.assert_not_called()
        self.assertFalse(self.ctx.sitting_event.is_set())

    def test_recovered_requires_sp_still_high(self) -> None:
        worker = self._worker(_ScriptedVitals([0.99, 0.02]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._seated = True
        worker.sit = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.danger.pop_damage_detected.return_value = False
        outcome = worker._sit_until_done(82)
        self.assertIsNone(outcome)


if __name__ == "__main__":
    unittest.main()
