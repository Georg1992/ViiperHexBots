"""Sit-on-low-SP worker: clear area, sit once, stand once, resume."""

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
    """Pops scripted SP ratios on each ``sp_pair`` read."""

    def __init__(self, ratios: list[float | None]) -> None:
        super().__init__()
        self._ratios = list(ratios)
        self.calls = 0

    def sp_pair(self) -> tuple[int | None, int | None]:
        self.calls += 1
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
        self.ctx.capture.capture_client.return_value = object()
        self.input = MagicMock(spec=ShadowInputBackend)
        from pybot.runtime.teleport import TeleportController
        self.teleport = TeleportController(self.ctx, self.input, MagicMock())
        self.teleport.teleport_until_quiet = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.danger = MagicMock(spec=DangerDetector)
        self.danger.pop_damage_detected.return_value = False

    def _worker(self, vitals: PlayerVitals) -> SitOnLowSpWorker:
        return SitOnLowSpWorker(
            self.ctx,
            self.input,
            self.teleport,
            danger=self.danger,
            vitals=vitals,
        )

    def _living_then_clear(self) -> list[DiscoveryScanResult]:
        living = RawDetection(
            x=10, y=10, confidence=0.9, candidate_scale=1.0, living=True,
            bbox=(0, 0, 20, 20),
        )
        empty = DiscoveryScanResult(
            ok=True,
            fail_reason="",
            raw_count=0,
            accepted_count=0,
            detections=[],
            duration_ms=1,
            elapsed_s=0.001,
        )
        living_scan = DiscoveryScanResult(
            ok=True,
            fail_reason="",
            raw_count=1,
            accepted_count=1,
            detections=[living],
            duration_ms=1,
            elapsed_s=0.001,
        )
        return [living_scan, empty, empty, empty, empty]

    def test_sitting_blocks_should_run_workers(self) -> None:
        self.assertTrue(self.ctx.should_run_workers())
        self.ctx.begin_sit_ops()
        self.assertFalse(self.ctx.should_run_workers())
        self.ctx.end_sit_ops()
        self.assertTrue(self.ctx.should_run_workers())

    def test_pose_settle_covers_sit_stand_animation(self) -> None:
        self.assertGreaterEqual(SIT_POSE_SETTLE_S, 0.3)

    def test_thresholds(self) -> None:
        self.assertAlmostEqual(SIT_LOW_SP_RATIO, 0.05)
        self.assertAlmostEqual(SIT_RESUME_SP_RATIO, 0.98)

    def test_recover_sits_once_stands_once_no_re_sit(self) -> None:
        """The bug: sit → stand → sit → hunt. Must be sit → stand → hunt."""
        vitals = _ScriptedVitals(
            [
                SIT_LOW_SP_RATIO - 0.01,
                0.50,
                SIT_RESUME_SP_RATIO,
            ]
        )
        self.ctx.detector.discover_frame.side_effect = self._living_then_clear()
        worker = self._worker(vitals)
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]
        # Sit: not sitting → after press sitting.
        # Stand (force): after press standing.
        # finally: _seated False → no extra press.
        worker._pose_is_sitting = MagicMock(  # type: ignore[method-assign]
            side_effect=[False, True, False]
        )

        def stop_after_recover() -> None:
            while self.input.key_tap.call_count < 2 and not self.ctx.is_stopped():
                self.ctx.stop_event.wait(0.01)
            self.ctx.stop_event.set()

        threading.Thread(target=stop_after_recover, daemon=True).start()
        worker.run()

        sit_presses = [
            c.args[0] for c in self.input.key_tap.call_args_list if c.args[0] == 82
        ]
        # Exactly one sit + one stand — never a third re-sit before hunt.
        self.assertEqual(len(sit_presses), 2, sit_presses)
        self.assertFalse(worker._seated)
        self.assertFalse(self.ctx.sitting_event.is_set())

    def test_finally_does_not_press_after_successful_stand(self) -> None:
        vitals = _ScriptedVitals([SIT_LOW_SP_RATIO - 0.01])
        worker = self._worker(vitals)
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]
        worker._sit_session = MagicMock(return_value="recovered")  # type: ignore[method-assign]
        worker._seated = False  # session already stood
        worker._recover_sp(SIT_LOW_SP_RATIO - 0.01)
        self.input.key_tap.assert_not_called()
        self.assertFalse(self.ctx.sitting_event.is_set())

    def test_finally_stands_when_still_seated(self) -> None:
        vitals = _ScriptedVitals([SIT_LOW_SP_RATIO - 0.01])
        worker = self._worker(vitals)
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]

        def abort_still_seated() -> None:
            worker._seated = True
            return None

        worker._sit_session = abort_still_seated  # type: ignore[method-assign]
        # force_press then confirm standing
        worker._pose_is_sitting = MagicMock(return_value=False)  # type: ignore[method-assign]
        worker._recover_sp(SIT_LOW_SP_RATIO - 0.01)
        self.input.key_tap.assert_called_once_with(82)
        self.assertFalse(worker._seated)
        self.assertFalse(self.ctx.sitting_event.is_set())

    def test_sit_teleport_clears_overlay_tracks(self) -> None:
        vitals = _ScriptedVitals([SIT_LOW_SP_RATIO - 0.01, SIT_RESUME_SP_RATIO])
        self.ctx.detector.discover_frame.side_effect = self._living_then_clear()
        worker = self._worker(vitals)
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]
        worker._pose_is_sitting = MagicMock(  # type: ignore[method-assign]
            side_effect=[False, True, False]
        )

        def stop_after_recover() -> None:
            while self.input.key_tap.call_count < 2 and not self.ctx.is_stopped():
                self.ctx.stop_event.wait(0.01)
            self.ctx.stop_event.set()

        threading.Thread(target=stop_after_recover, daemon=True).start()
        worker.run()
        self.teleport.teleport_until_quiet.assert_called_once()

    def test_sit_session_interrupted_stands_then_returns(self) -> None:
        vitals = _ScriptedVitals([0.40] * 20)
        worker = self._worker(vitals)
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]
        # Sit confirm; force stand press then standing.
        worker._pose_is_sitting = MagicMock(  # type: ignore[method-assign]
            side_effect=[True, False]
        )
        self.danger.pop_damage_detected.side_effect = [False, True]
        outcome = worker._sit_session()

        self.assertEqual(outcome, "interrupted")
        self.input.key_tap.assert_called_once_with(82)
        self.assertFalse(worker._seated)

    def test_sit_skips_press_when_already_sitting(self) -> None:
        worker = self._worker(_ScriptedVitals([]))
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]
        worker._pose_is_sitting = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.assertTrue(worker._sit(82))
        self.input.key_tap.assert_not_called()
        self.assertTrue(worker._seated)

    def test_stand_if_seated_force_presses_even_if_pose_looks_standing(self) -> None:
        """Falcon can make sitting look standing — seated flag still presses once."""
        worker = self._worker(_ScriptedVitals([]))
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]
        worker._seated = True
        worker._pose_is_sitting = MagicMock(return_value=False)  # type: ignore[method-assign]
        self.assertTrue(worker._stand_if_seated(82))
        self.input.key_tap.assert_called_once_with(82)
        self.assertFalse(worker._seated)

    def test_stand_if_seated_is_noop_when_not_seated(self) -> None:
        worker = self._worker(_ScriptedVitals([]))
        worker._seated = False
        self.assertTrue(worker._stand_if_seated(82))
        self.input.key_tap.assert_not_called()

    def test_ensure_sitting_presses_when_not_sitting(self) -> None:
        worker = self._worker(_ScriptedVitals([]))
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]
        worker._pose_is_sitting = MagicMock(  # type: ignore[method-assign]
            side_effect=[False, True]
        )
        self.assertTrue(worker._sit(82))
        self.input.key_tap.assert_called_once_with(82)

    def test_unreadable_pose_does_not_count_as_standing(self) -> None:
        worker = self._worker(_ScriptedVitals([]))
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]
        worker._seated = True
        # force press → None → retry press → None → press → standing
        worker._pose_is_sitting = MagicMock(  # type: ignore[method-assign]
            side_effect=[None, None, None, False]
        )
        self.assertTrue(worker._stand_if_seated(82))
        self.assertGreaterEqual(self.input.key_tap.call_count, 1)
        self.assertFalse(worker._seated)

    def test_reach_pose_waits_settle_after_press_before_verify(self) -> None:
        worker = self._worker(_ScriptedVitals([]))
        waits: list[float] = []

        def record_wait(timeout_s: float) -> bool:
            waits.append(timeout_s)
            return True

        self.ctx.wait_unless_stopped = record_wait  # type: ignore[method-assign]
        worker._pose_is_sitting = MagicMock(  # type: ignore[method-assign]
            side_effect=[False, True]
        )
        self.assertTrue(worker._sit(82))
        self.input.key_tap.assert_called_once_with(82)
        self.assertEqual(waits, [SIT_POSE_SETTLE_S])

    def test_stand_failure_keeps_sit_gate_and_retries(self) -> None:
        vitals = _ScriptedVitals([SIT_RESUME_SP_RATIO, SIT_RESUME_SP_RATIO])
        worker = self._worker(vitals)
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]
        worker._sit = MagicMock(return_value=True)  # type: ignore[method-assign]
        worker._seated = True
        worker._stand_if_seated = MagicMock(return_value=False)  # type: ignore[method-assign]
        self.danger.pop_damage_detected.return_value = False

        def stop_soon(*_a, **_k):
            self.ctx.stop_event.set()
            return False

        self.ctx.stop_event.wait = stop_soon  # type: ignore[method-assign]
        outcome = worker._sit_session()
        self.assertIsNone(outcome)
        worker._stand_if_seated.assert_called()

    def test_recover_stands_before_releasing_hunt_gate(self) -> None:
        vitals = _ScriptedVitals([SIT_LOW_SP_RATIO - 0.01])
        worker = self._worker(vitals)
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]

        def abort_seated() -> None:
            worker._seated = True
            return None

        worker._sit_session = abort_seated  # type: ignore[method-assign]
        gate_during_stand: list[bool] = []

        def stand(scan: int) -> bool:
            gate_during_stand.append(self.ctx.sitting_event.is_set())
            worker._seated = False
            return True

        worker._stand_if_seated = stand  # type: ignore[method-assign]
        worker._recover_sp(SIT_LOW_SP_RATIO - 0.01)
        self.assertTrue(all(gate_during_stand))
        self.assertFalse(self.ctx.sitting_event.is_set())
        self.assertTrue(self.ctx.should_run_combat())

    def test_pause_during_sit_keeps_gate_until_resume(self) -> None:
        vitals = _ScriptedVitals([0.40] * 30)
        worker = self._worker(vitals)
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]
        worker._pose_is_sitting = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.danger.pop_damage_detected.return_value = False
        self.ctx.begin_sit_ops()
        self.ctx.pause_event.set()

        outcomes: list[str | None] = []

        def run_session() -> None:
            outcomes.append(worker._sit_session())

        thread = threading.Thread(target=run_session, daemon=True)
        thread.start()
        self.ctx.stop_event.wait(0.15)
        self.assertTrue(self.ctx.sitting_event.is_set())
        self.assertFalse(self.ctx.should_run_workers())
        self.assertTrue(thread.is_alive())
        self.ctx.pause_event.clear()
        self.ctx.stop_event.wait(0.1)
        self.assertTrue(thread.is_alive())
        self.assertTrue(self.ctx.sitting_event.is_set())
        self.ctx.stop_event.set()
        thread.join(timeout=1.0)
        self.assertEqual(outcomes, [None])


if __name__ == "__main__":
    unittest.main()
