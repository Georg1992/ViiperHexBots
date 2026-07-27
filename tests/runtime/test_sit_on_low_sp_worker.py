"""Sit-on-low-SP worker: clear area, sit, wait, stand, resume."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import MagicMock, patch

from pybot.game_state import PlayerVitals
from pybot.recognition.danger import DangerReport
from pybot.recognition.ui.character_pose import CharacterPose
from pybot.runtime.constants import SIT_LOW_SP_RATIO, SIT_RESUME_SP_RATIO
from pybot.runtime.detection.detector_session import DiscoveryScanResult, RawDetection
from pybot.runtime.input.input_backend import ShadowInputBackend
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.workers.sit_on_low_sp_worker import SitOnLowSpWorker

_STAND = CharacterPose(body_height=99, fg_count=2500)
_SIT = CharacterPose(body_height=60, fg_count=2200)


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
        self.config.active_teleport_scan_code.return_value = 16
        self.config.active_teleport_button.return_value = "q"
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
        self.hunt_mode = MagicMock()

    def _empty_scans(self, n: int = 5) -> list[DiscoveryScanResult]:
        empty = DiscoveryScanResult(
            ok=True,
            fail_reason="",
            raw_count=0,
            accepted_count=0,
            detections=[],
            duration_ms=1,
            elapsed_s=0.001,
        )
        return [empty] * n

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

    def _pose_side_effect(self, poses: list[CharacterPose]):
        it = iter(poses)

        def _next() -> CharacterPose:
            try:
                return next(it)
            except StopIteration:
                return poses[-1]

        return _next

    def test_sitting_blocks_should_run_workers(self) -> None:
        self.assertTrue(self.ctx.should_run_workers())
        self.ctx.begin_sit_regen()
        self.assertFalse(self.ctx.should_run_workers())
        self.ctx.end_sit_regen()
        self.assertTrue(self.ctx.should_run_workers())

    def test_recover_teleports_until_clear_then_sits(self) -> None:
        vitals = _ScriptedVitals(
            [
                SIT_LOW_SP_RATIO - 0.01,
                0.50,
                SIT_RESUME_SP_RATIO,
            ]
        )
        self.ctx.detector.discover_frame.side_effect = self._living_then_clear()
        worker = SitOnLowSpWorker(
            self.ctx,
            self.input,
            hunt_mode=self.hunt_mode,
            vitals=vitals,
        )
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]
        # stand calibrate, sit confirm, stand check (still sit), stand after press
        poses = [_STAND, _SIT, _SIT, _STAND]

        def stop_after_recover() -> None:
            while self.input.teleport_key.call_count < 3 and not self.ctx.is_stopped():
                self.ctx.stop_event.wait(0.01)
            self.ctx.stop_event.set()

        threading.Thread(target=stop_after_recover, daemon=True).start()
        with patch.object(worker, "_measure_pose", side_effect=self._pose_side_effect(poses)):
            worker.run()

        self.assertGreaterEqual(self.input.teleport_key.call_count, 3)
        self.assertEqual(self.input.teleport_key.call_args_list[0].args[0], 16)
        sit_presses = [
            c.args[0] for c in self.input.teleport_key.call_args_list if c.args[0] == 82
        ]
        self.assertGreaterEqual(len(sit_presses), 2)
        self.assertFalse(self.ctx.sitting_event.is_set())
        self.assertTrue(self.ctx.discovery_wake.is_set())
        self.assertGreaterEqual(self.ctx.tracks.area_reset.call_count, 1)
        self.assertGreaterEqual(self.hunt_mode.on_area_reset.call_count, 1)

    def test_thresholds(self) -> None:
        self.assertAlmostEqual(SIT_LOW_SP_RATIO, 0.05)
        self.assertAlmostEqual(SIT_RESUME_SP_RATIO, 0.98)

    def test_sit_teleport_clears_overlay_tracks(self) -> None:
        vitals = _ScriptedVitals([SIT_LOW_SP_RATIO - 0.01, SIT_RESUME_SP_RATIO])
        self.ctx.detector.discover_frame.side_effect = self._living_then_clear()
        worker = SitOnLowSpWorker(
            self.ctx,
            self.input,
            hunt_mode=self.hunt_mode,
            vitals=vitals,
        )
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]
        poses = [_STAND, _SIT, _SIT, _STAND]

        def stop_after_recover() -> None:
            while self.input.teleport_key.call_count < 3 and not self.ctx.is_stopped():
                self.ctx.stop_event.wait(0.01)
            self.ctx.stop_event.set()

        threading.Thread(target=stop_after_recover, daemon=True).start()
        with patch.object(worker, "_measure_pose", side_effect=self._pose_side_effect(poses)):
            worker.run()

        self.ctx.overlay.set_track_positions.assert_called_with([])
        self.ctx.overlay.set_track_stats.assert_any_call(track_count=0, alive_count=0)

    def test_sit_session_returns_danger_on_sp_drop_and_near_objects(self) -> None:
        # SP drop marks stall immediately; keep mid ratios so we never resume.
        vitals = _ScriptedVitals([0.40, 0.30] + [0.30] * 30)
        worker = SitOnLowSpWorker(
            self.ctx,
            self.input,
            hunt_mode=self.hunt_mode,
            vitals=vitals,
        )
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]
        # stand, sit confirm, then standing checks during ensure_standing
        poses = [_STAND, _SIT, _STAND]

        with patch(
            "pybot.runtime.workers.sit_on_low_sp_worker.SIT_HP_POLL_S",
            0.0,
        ):
            with patch.object(worker, "_measure_pose", side_effect=self._pose_side_effect(poses)):
                with patch.object(worker, "_capture_client", return_value=object()):
                    with patch.object(worker, "_read_hp", return_value=1000):
                        with patch.object(
                            worker,
                            "_assess_danger",
                            return_value=DangerReport(
                                in_danger=True,
                                reasons=("near_objects:1",),
                                near_object_count=1,
                            ),
                        ):
                            outcome = worker._sit_session()

        self.assertEqual(outcome, "danger")
        self.assertGreaterEqual(self.input.teleport_key.call_count, 1)

    def test_sit_session_returns_danger_on_hp_drop(self) -> None:
        # Steady SP mid-regen; danger comes from HP drop only.
        # Enough mid ratios so scripted vitals never fall through to 98% resume.
        vitals = _ScriptedVitals([0.40] * 20)
        worker = SitOnLowSpWorker(
            self.ctx,
            self.input,
            hunt_mode=self.hunt_mode,
            vitals=vitals,
        )
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]
        poses = [_STAND, _SIT]
        hp_values = iter([1000, 900])

        def fake_assess(frame, *, hp=None, previous_hp=None):
            del frame
            if (
                hp is not None
                and previous_hp is not None
                and hp < previous_hp
            ):
                return DangerReport(
                    in_danger=True,
                    reasons=(f"hp_drop:{previous_hp}->{hp}",),
                    near_object_count=0,
                )
            return DangerReport(in_danger=False, reasons=(), near_object_count=0)

        with patch(
            "pybot.runtime.workers.sit_on_low_sp_worker.SIT_HP_POLL_S",
            0.0,
        ):
            with patch.object(
                worker, "_measure_pose", side_effect=self._pose_side_effect(poses)
            ):
                with patch.object(worker, "_capture_client", return_value=object()):
                    with patch.object(
                        worker, "_read_hp", side_effect=lambda _f: next(hp_values, 900)
                    ):
                        with patch.object(
                            worker, "_assess_danger", side_effect=fake_assess
                        ):
                            outcome = worker._sit_session()

        self.assertEqual(outcome, "danger")

    def test_ensure_sitting_retries_until_pose_drops(self) -> None:
        worker = SitOnLowSpWorker(
            self.ctx,
            self.input,
            hunt_mode=self.hunt_mode,
            vitals=_ScriptedVitals([]),
        )
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]
        # First press still looks standing; second press sits.
        poses = [_STAND, _SIT]
        with patch.object(worker, "_measure_pose", side_effect=self._pose_side_effect(poses)):
            sit_pose = worker._ensure_sitting(82, _STAND)
        self.assertIsNotNone(sit_pose)
        assert sit_pose is not None
        self.assertEqual(sit_pose.body_height, _SIT.body_height)
        self.assertEqual(self.input.teleport_key.call_count, 2)

    def test_ensure_standing_retries_until_pose_rises(self) -> None:
        worker = SitOnLowSpWorker(
            self.ctx,
            self.input,
            hunt_mode=self.hunt_mode,
            vitals=_ScriptedVitals([]),
        )
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]
        # Pre-check sit → press → still sit → loop → pre-check sit → press → stand.
        poses = [_SIT, _SIT, _SIT, _STAND]
        with patch.object(worker, "_measure_pose", side_effect=self._pose_side_effect(poses)):
            ok = worker._ensure_standing(82, _SIT, _STAND)
        self.assertTrue(ok)
        self.assertEqual(self.input.teleport_key.call_count, 2)

    def test_ensure_sitting_fails_after_max_attempts(self) -> None:
        worker = SitOnLowSpWorker(
            self.ctx,
            self.input,
            hunt_mode=self.hunt_mode,
            vitals=_ScriptedVitals([]),
        )
        self.ctx.wait_unless_stopped = lambda _timeout_s: True  # type: ignore[method-assign]
        with patch.object(worker, "_measure_pose", return_value=_STAND):
            sit_pose = worker._ensure_sitting(82, _STAND)
        self.assertIsNone(sit_pose)
        self.assertEqual(self.input.teleport_key.call_count, 5)


if __name__ == "__main__":
    unittest.main()
