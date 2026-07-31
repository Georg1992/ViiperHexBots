"""Sit-on-low-SP: sit once, wait, stand once, resume — never re-sit."""

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

    def _worker(self, vitals: PlayerVitals) -> SitOnLowSpWorker:
        return SitOnLowSpWorker(
            self.ctx, self.input, self.teleport,
            danger=self.danger, vitals=vitals,
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

    def test_sitting_blocks_hunt(self) -> None:
        self.assertTrue(self.ctx.should_run_workers())
        self.ctx.begin_sit_ops()
        self.assertFalse(self.ctx.should_run_workers())
        self.ctx.end_sit_ops()
        self.assertTrue(self.ctx.should_run_workers())

    def test_happy_path_exactly_one_sit_and_one_stand(self) -> None:
        vitals = _ScriptedVitals(
            [SIT_LOW_SP_RATIO - 0.01, 0.50, SIT_RESUME_SP_RATIO]
        )
        self.ctx.detector.discover_frame.side_effect = self._living_then_clear()
        worker = self._worker(vitals)
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        # sit: not sitting → after tap sitting; stand: after tap standing
        worker._pose_is_sitting = MagicMock(side_effect=[False, True, False])  # type: ignore[method-assign]

        def stop_after_two_taps() -> None:
            while self.input.key_tap.call_count < 2 and not self.ctx.is_stopped():
                self.ctx.stop_event.wait(0.01)
            self.ctx.stop_event.set()

        threading.Thread(target=stop_after_two_taps, daemon=True).start()
        worker.run()

        presses = [c.args[0] for c in self.input.key_tap.call_args_list if c.args[0] == 82]
        self.assertEqual(len(presses), 2, presses)
        self.assertFalse(worker._seated)
        self.assertFalse(self.ctx.sitting_event.is_set())

    def test_finally_noop_after_successful_stand(self) -> None:
        worker = self._worker(_ScriptedVitals([SIT_LOW_SP_RATIO - 0.01]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._sit_until_done = MagicMock(return_value="recovered")  # type: ignore[method-assign]
        worker._seated = False
        worker._recover_sp(SIT_LOW_SP_RATIO - 0.01)
        self.input.key_tap.assert_not_called()
        self.assertFalse(self.ctx.sitting_event.is_set())

    def test_finally_stands_when_still_seated(self) -> None:
        worker = self._worker(_ScriptedVitals([SIT_LOW_SP_RATIO - 0.01]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._sit_until_done = MagicMock(side_effect=lambda _s: setattr(worker, "_seated", True) or None)  # type: ignore[method-assign]
        worker._pose_is_sitting = MagicMock(return_value=False)  # type: ignore[method-assign]
        worker._recover_sp(SIT_LOW_SP_RATIO - 0.01)
        self.input.key_tap.assert_called_once_with(82)
        self.assertFalse(worker._seated)

    def test_stand_always_presses_when_seated(self) -> None:
        worker = self._worker(_ScriptedVitals([]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._seated = True
        worker._pose_is_sitting = MagicMock(return_value=False)  # type: ignore[method-assign]
        self.assertTrue(worker.stand(82))
        self.input.key_tap.assert_called_once_with(82)
        self.assertFalse(worker._seated)

    def test_stand_noop_when_not_seated(self) -> None:
        worker = self._worker(_ScriptedVitals([]))
        worker._seated = False
        self.assertTrue(worker.stand(82))
        self.input.key_tap.assert_not_called()

    def test_sit_skips_press_when_already_sitting(self) -> None:
        worker = self._worker(_ScriptedVitals([]))
        worker._pose_is_sitting = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.assertTrue(worker.sit(82))
        self.input.key_tap.assert_not_called()
        self.assertTrue(worker._seated)

    def test_sit_presses_when_standing(self) -> None:
        worker = self._worker(_ScriptedVitals([]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._pose_is_sitting = MagicMock(side_effect=[False, True])  # type: ignore[method-assign]
        self.assertTrue(worker.sit(82))
        self.input.key_tap.assert_called_once_with(82)
        self.assertTrue(worker._seated)

    def test_unreadable_pose_is_not_standing(self) -> None:
        worker = self._worker(_ScriptedVitals([]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._seated = True
        worker._pose_is_sitting = MagicMock(side_effect=[None, None, False])  # type: ignore[method-assign]
        self.assertTrue(worker.stand(82))
        self.assertGreaterEqual(self.input.key_tap.call_count, 1)
        self.assertFalse(worker._seated)

    def test_tap_waits_pose_settle(self) -> None:
        worker = self._worker(_ScriptedVitals([]))
        waits: list[float] = []
        self.ctx.wait_unless_stopped = lambda t: waits.append(t) or True  # type: ignore[method-assign]
        worker._pose_is_sitting = MagicMock(side_effect=[False, True])  # type: ignore[method-assign]
        self.assertTrue(worker.sit(82))
        self.assertEqual(waits, [SIT_POSE_SETTLE_S])

    def test_damage_interrupt_stands_once(self) -> None:
        worker = self._worker(_ScriptedVitals([0.40] * 20))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._pose_is_sitting = MagicMock(side_effect=[True, False])  # type: ignore[method-assign]
        self.danger.pop_damage_detected.side_effect = [False, True]
        self.assertEqual(worker._sit_until_done(82), "interrupted")
        self.input.key_tap.assert_called_once_with(82)
        self.assertFalse(worker._seated)

    def test_stand_failure_keeps_gate(self) -> None:
        worker = self._worker(_ScriptedVitals([SIT_RESUME_SP_RATIO] * 4))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker.sit = MagicMock(return_value=True)  # type: ignore[method-assign]
        worker._seated = True
        worker.stand = MagicMock(return_value=False)  # type: ignore[method-assign]

        def stop_soon(*_a, **_k):
            self.ctx.stop_event.set()
            return False

        self.ctx.stop_event.wait = stop_soon  # type: ignore[method-assign]
        self.assertIsNone(worker._sit_until_done(82))
        worker.stand.assert_called()

    def test_gate_held_during_final_stand(self) -> None:
        worker = self._worker(_ScriptedVitals([SIT_LOW_SP_RATIO - 0.01]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._sit_until_done = MagicMock(side_effect=lambda _s: setattr(worker, "_seated", True) or None)  # type: ignore[method-assign]
        held: list[bool] = []

        def stand(_scan: int) -> bool:
            held.append(self.ctx.sitting_event.is_set())
            worker._seated = False
            return True

        worker.stand = stand  # type: ignore[method-assign]
        worker._recover_sp(SIT_LOW_SP_RATIO - 0.01)
        self.assertTrue(all(held))
        self.assertFalse(self.ctx.sitting_event.is_set())

    def test_pause_during_sit_keeps_gate(self) -> None:
        worker = self._worker(_ScriptedVitals([0.40] * 30))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._pose_is_sitting = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.ctx.begin_sit_ops()
        self.ctx.pause_event.set()
        outcomes: list[str | None] = []

        def run() -> None:
            outcomes.append(worker._sit_until_done(82))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.ctx.stop_event.wait(0.15)
        self.assertTrue(thread.is_alive())
        self.assertTrue(self.ctx.sitting_event.is_set())
        self.ctx.pause_event.clear()
        self.ctx.stop_event.wait(0.1)
        self.assertTrue(thread.is_alive())
        self.ctx.stop_event.set()
        thread.join(timeout=1.0)
        self.assertEqual(outcomes, [None])


if __name__ == "__main__":
    unittest.main()
