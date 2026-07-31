"""Sit must hold hunt until SP recovers; toggle must not re-sit."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import MagicMock

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import (
    SIT_LOW_SP_RATIO,
    SIT_POSE_SETTLE_S,
    SIT_RESUME_SP_RATIO,
)
from pybot.runtime.danger_detector import DangerDetector
from pybot.runtime.detection.detector_session import DiscoveryScanResult, RawDetection
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
        self.ctx.capture.is_valid.return_value = True
        self.ctx.capture.get_hunt_roi.return_value = MagicMock(x=0, y=0, w=100, h=100)
        self.ctx.capture.capture_roi.return_value = MagicMock(size=1)
        self.input = MagicMock(spec=ShadowInputBackend)
        from pybot.runtime.teleport import TeleportController
        self.teleport = TeleportController(self.ctx, self.input, MagicMock())
        self.teleport.teleport_until_quiet = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.danger = MagicMock(spec=DangerDetector)
        self.danger.pop_damage_detected.return_value = False

    def _worker(self, vitals: PlayerVitals | None = None) -> SitOnLowSpWorker:
        return SitOnLowSpWorker(
            self.ctx, self.input, self.teleport,
            danger=self.danger, vitals=vitals or _ScriptedVitals([]),
        )

    def _living_then_clear(self) -> list[DiscoveryScanResult]:
        living = RawDetection(
            x=10, y=10, confidence=0.9, candidate_scale=1.0, living=True,
            bbox=(0, 0, 20, 20),
        )
        empty = DiscoveryScanResult(
            ok=True, fail_reason="", raw_count=0, accepted_count=0,
            detections=[], duration_ms=1, elapsed_s=0.001,
        )
        living_scan = DiscoveryScanResult(
            ok=True, fail_reason="", raw_count=1, accepted_count=1,
            detections=[living], duration_ms=1, elapsed_s=0.001,
        )
        return [living_scan, empty, empty, empty, empty]

    def test_thresholds(self) -> None:
        self.assertAlmostEqual(SIT_LOW_SP_RATIO, 0.05)
        self.assertAlmostEqual(SIT_RESUME_SP_RATIO, 0.98)
        self.assertGreaterEqual(SIT_POSE_SETTLE_S, 0.3)

    def test_failed_clear_does_not_resume_hunt_while_sp_low(self) -> None:
        """Bug: teleport_until_quiet False → end_sit_ops → hunt on empty SP."""
        vitals = _ScriptedVitals([SIT_LOW_SP_RATIO - 0.01] * 50)
        worker = self._worker(vitals)
        self.teleport.teleport_until_quiet = MagicMock(return_value=False)  # type: ignore[method-assign]
        clear_calls = {"n": 0}

        def clear(**_k) -> bool:
            clear_calls["n"] += 1
            if clear_calls["n"] >= 3:
                self.ctx.stop_event.set()
            return False

        self.teleport.teleport_until_quiet = clear  # type: ignore[method-assign]
        worker._recover_sp(SIT_LOW_SP_RATIO - 0.01)
        self.assertGreaterEqual(clear_calls["n"], 3)
        # After stop, gate released — but while retrying it must have been held.
        self.assertGreaterEqual(
            self.ctx.logger.behavior.call_count, 1
        )
        self.assertTrue(
            any(
                "hunt stays paused" in str(c)
                for c in self.ctx.logger.behavior.call_args_list
            )
        )

    def test_failed_sit_retries_without_resuming_hunt(self) -> None:
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
        self.assertTrue(
            any(
                "hunt stays paused" in str(c)
                for c in self.ctx.logger.behavior.call_args_list
            )
        )

    def test_recovered_requires_sp_still_high(self) -> None:
        worker = self._worker(_ScriptedVitals([0.99, 0.02]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._seated = True
        worker.sit = MagicMock(return_value=True)  # type: ignore[method-assign]
        worker.stand = MagicMock(side_effect=lambda _s: setattr(worker, "_seated", False) or True)  # type: ignore[method-assign]
        self.danger.pop_damage_detected.return_value = False
        # First SP read in loop = 0.99 → stand; after stand SP = 0.02 → not recovered
        outcome = worker._sit_until_done(82)
        self.assertIsNone(outcome)

    def test_happy_path_one_sit_one_stand(self) -> None:
        vitals = _ScriptedVitals(
            [SIT_LOW_SP_RATIO - 0.01, 0.50, SIT_RESUME_SP_RATIO, SIT_RESUME_SP_RATIO]
        )
        self.ctx.detector.discover_frame.side_effect = self._living_then_clear()
        worker = self._worker(vitals)
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._pose_is_sitting = MagicMock(side_effect=[False, True, False])  # type: ignore[method-assign]

        def stop_after_two() -> None:
            while self.input.key_tap.call_count < 2 and not self.ctx.is_stopped():
                self.ctx.stop_event.wait(0.01)
            self.ctx.stop_event.set()

        threading.Thread(target=stop_after_two, daemon=True).start()
        worker.run()
        presses = [c.args[0] for c in self.input.key_tap.call_args_list if c.args[0] == 82]
        self.assertEqual(len(presses), 2, presses)
        self.assertFalse(worker._seated)

    def test_stand_does_not_second_press_on_ambiguous_pose(self) -> None:
        worker = self._worker()
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._seated = True
        worker._pose_is_sitting = MagicMock(side_effect=[None, None, False])  # type: ignore[method-assign]
        self.assertTrue(worker.stand(82))
        self.assertEqual(self.input.key_tap.call_count, 1)
        self.assertFalse(worker._seated)

    def test_stand_second_press_only_if_confirmed_still_sitting(self) -> None:
        worker = self._worker()
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._seated = True
        worker._pose_is_sitting = MagicMock(side_effect=[True, False])  # type: ignore[method-assign]
        self.assertTrue(worker.stand(82))
        self.assertEqual(self.input.key_tap.call_count, 2)

    def test_stand_noop_when_not_seated(self) -> None:
        worker = self._worker()
        worker._seated = False
        self.assertTrue(worker.stand(82))
        self.input.key_tap.assert_not_called()

    def test_sit_skips_press_when_already_sitting(self) -> None:
        worker = self._worker()
        worker._pose_is_sitting = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.assertTrue(worker.sit(82))
        self.input.key_tap.assert_not_called()
        self.assertTrue(worker._seated)

    def test_finally_stands_before_release_after_recover(self) -> None:
        # Default vitals after script empties → 98% (recovered).
        worker = self._worker(_ScriptedVitals([SIT_LOW_SP_RATIO - 0.01]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._sit_until_done = MagicMock(return_value="recovered")  # type: ignore[method-assign]
        worker._seated = False
        worker._recover_sp(SIT_LOW_SP_RATIO - 0.01)
        self.input.key_tap.assert_not_called()
        self.assertFalse(self.ctx.sitting_event.is_set())

    def test_damage_interrupt_stands(self) -> None:
        worker = self._worker(_ScriptedVitals([0.40] * 20))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._pose_is_sitting = MagicMock(side_effect=[True, False])  # type: ignore[method-assign]
        self.danger.pop_damage_detected.side_effect = [False, True]
        self.assertEqual(worker._sit_until_done(82), "interrupted")
        self.input.key_tap.assert_called_once_with(82)
        self.assertFalse(worker._seated)


if __name__ == "__main__":
    unittest.main()
