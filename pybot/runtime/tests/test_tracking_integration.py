"""LocalTracker + ConfirmState integration on fixture frames."""

from __future__ import annotations

import threading
import unittest

import cv2
import numpy as np

from pybot.runtime.capture.window_roi import HuntRoi
from pybot.runtime.config import PROJECT_ROOT, HuntRuntimeConfig
from pybot.runtime.control import RuntimeControl
from pybot.runtime.hunt_mode import create_hunt_mode
from pybot.runtime.hunt_policy import HuntPolicy
from pybot.runtime.hunt_tracks import HuntTracks, monotonic_ms
from pybot.runtime.input.input_backend import ShadowInputBackend
from pybot.runtime.logging import HuntLogger
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.urgent_state import UrgentStateQueue
from pybot.runtime.validation_log import HuntValidationLogger
from pybot.runtime.detection.detector_session import DetectorSession, StateTrackSnapshot
from pybot.runtime.workers.attack_loop import AttackLoop
from pybot.runtime.workers.confirm_state_worker import ConfirmStateWorker
from pybot.runtime.workers.discovery_worker import DiscoveryWorker
from pybot.runtime.workers.tracking_worker import TrackingWorker

from pybot.runtime._mob_rec_path import import_hunt_track_rules
_hunt = import_hunt_track_rules()
HUNT_LOCAL_TRACK_MISS_LIMIT = _hunt.HUNT_LOCAL_TRACK_MISS_LIMIT

FIXTURE = PROJECT_ROOT / "mob-recognition" / "test-fixtures" / "game-screenshots" / "333.png"


def playfield_roi(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


class FixtureDetector(DetectorSession):
    def __init__(self, frame: np.ndarray) -> None:
        super().__init__("horn", project_root=PROJECT_ROOT)
        self._fixture_frame = frame

    def discover(self, roi: HuntRoi):
        return self.discover_frame(self._fixture_frame, roi)

    def track_locals(self, roi: HuntRoi, track_snapshots):
        return self.track_locals_frame(self._fixture_frame, roi, track_snapshots)

    def state_direct(self, roi: HuntRoi, snapshot: StateTrackSnapshot):
        return self.state_direct_frame(self._fixture_frame, roi, snapshot)

    def state_confirm(self, roi: HuntRoi, snapshot: StateTrackSnapshot):
        return self.state_confirm_frame(self._fixture_frame, roi, snapshot)


class FakeCapture:
    def __init__(self, roi: HuntRoi) -> None:
        self._roi = roi

    def is_valid(self) -> bool:
        return True

    def get_hunt_roi(self) -> HuntRoi:
        return self._roi


def make_config(**overrides) -> HuntRuntimeConfig:
    base = dict(
        config_path=PROJECT_ROOT / "config.ini",
        hwnd=12345,
        mob_name="horn",
        hunt_mode="teleport",
        skill_delay_ms=500,
        skill_button="e",
        skill_scan_code=18,
        teleport_button="q",
        teleport_scan_code=16,
        search_range_cells=16,
        cell_size_px=64,
        state_interval_ms=100,
        discovery_interval_ms=3000,
        post_attack_state_delay_ms=0,
        teleport_duration_ms=500,
        coord_stale_skip_ms=None,
        validation_enabled=False,
        validation_state_every_n=1,
        control_file=None,
    )
    base.update(overrides)
    return HuntRuntimeConfig(**base)


def make_context(
    config: HuntRuntimeConfig,
    *,
    roi: HuntRoi,
    detector: DetectorSession,
) -> HuntRuntimeContext:
    logger = HuntLogger(session_id="test_tracking_integration")
    tracks = HuntTracks()
    stop = threading.Event()
    stop.set()
    return HuntRuntimeContext(
        config=config,
        logger=logger,
        tracks=tracks,
        policy=HuntPolicy(),
        capture=FakeCapture(roi),
        detector=detector,
        urgent=UrgentStateQueue(),
        validation=HuntValidationLogger(logger, tracks, enabled=False),
        control=RuntimeControl(None),
        stop_event=stop,
    )


class TrackingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        frame = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
        if frame is None:
            raise unittest.SkipTest("fixture 333.png missing")
        cls.roi_frame = playfield_roi(frame)
        cls.roi = HuntRoi(x=0, y=0, w=cls.roi_frame.shape[1], h=cls.roi_frame.shape[0])

    def test_local_track_batch_updates_all_alive_tracks(self) -> None:
        config = make_config()
        detector = FixtureDetector(self.roi_frame)
        ctx = make_context(config, roi=self.roi, detector=detector)
        hunt_mode = create_hunt_mode(ctx, ShadowInputBackend())

        DiscoveryWorker(ctx, hunt_mode)._run_scan()
        self.assertGreater(ctx.tracks.get_track_count(), 0)

        before = {
            track.id: (track.x, track.y, track.updated_tick)
            for track in ctx.tracks.tracks_for_policy(monotonic_ms())
        }

        TrackingWorker(ctx)._run_local_track_batch()

        after_tick = monotonic_ms()
        for track in ctx.tracks.tracks_for_policy(after_tick):
            self.assertIn(track.id, before)
            self.assertGreaterEqual(track.updated_tick, before[track.id][2])

    def test_shadow_attack_then_direct_confirm(self) -> None:
        config = make_config(post_attack_state_delay_ms=0, skill_delay_ms=0)
        detector = FixtureDetector(self.roi_frame)
        ctx = make_context(config, roi=self.roi, detector=detector)
        hunt_mode = create_hunt_mode(ctx, ShadowInputBackend())
        attack = AttackLoop(ctx, hunt_mode, ShadowInputBackend())
        confirm = ConfirmStateWorker(ctx)

        DiscoveryWorker(ctx, hunt_mode)._run_scan()
        TrackingWorker(ctx)._run_local_track_batch()

        now = monotonic_ms()
        target_id = ctx.policy.select_target(ctx.tracks.tracks_for_policy(now), now)
        self.assertIsNotNone(target_id)
        assert target_id is not None

        track_before = ctx.tracks.get_track_by_id(target_id)
        assert track_before is not None
        self.assertEqual(track_before.state, "alive")
        self.assertEqual(track_before.attack_count, 0)

        attack._handle_target(target_id, now)

        track_pending = ctx.tracks.get_track_by_id(target_id)
        assert track_pending is not None
        self.assertEqual(track_pending.state, "pending")
        self.assertEqual(track_pending.attack_count, 1)
        self.assertTrue(ctx.urgent.has_pending())

        self.assertTrue(confirm._run_urgent_direct())
        self.assertFalse(ctx.urgent.has_pending())

        track_after = ctx.tracks.get_track_by_id(target_id)
        if track_after is not None:
            self.assertIn(track_after.state, ("alive", "dead", "gone"))
        else:
            self.assertEqual(track_before.attack_count, 1)

    def test_local_miss_streak_schedules_confirm(self) -> None:
        config = make_config()
        detector = FixtureDetector(self.roi_frame)
        ctx = make_context(config, roi=self.roi, detector=detector)
        hunt_mode = create_hunt_mode(ctx, ShadowInputBackend())
        tracking = TrackingWorker(ctx)
        confirm = ConfirmStateWorker(ctx)

        DiscoveryWorker(ctx, hunt_mode)._run_scan()
        track = ctx.tracks.get_track_by_id(1)
        assert track is not None
        track.x = -5000
        track.y = -5000

        now = monotonic_ms()
        for _ in range(HUNT_LOCAL_TRACK_MISS_LIMIT):
            tracking._run_local_track_batch()
            now = monotonic_ms()

        missed = ctx.tracks.get_track_by_id(1)
        assert missed is not None
        self.assertGreaterEqual(missed.local_track_miss_count, HUNT_LOCAL_TRACK_MISS_LIMIT)

        confirm_id = ctx.tracks.select_state_confirm_track_id(now)
        self.assertEqual(confirm_id, 1)
        self.assertTrue(confirm._run_state_confirm())

        after = ctx.tracks.get_track_by_id(1)
        assert after is not None
        self.assertEqual(after.local_track_miss_count, 0)
        self.assertIn(after.state, ("alive", "dead", "gone"))


if __name__ == "__main__":
    unittest.main()
