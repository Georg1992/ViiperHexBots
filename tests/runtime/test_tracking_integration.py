"""Tracking + discovery integration on fixture frames."""

from __future__ import annotations

import threading
import unittest

import cv2
import numpy as np

from pybot.runtime.capture.window_roi import HuntRoi
from pybot.paths import PROJECT_ROOT, RECOGNITION_FIXTURES_DIR
from pybot.recognition.rules import DiscoveryDetection
from pybot.runtime.config import HuntRuntimeConfig
from pybot.runtime.control import RuntimeControl
from pybot.runtime.hunt_mode import create_hunt_mode
from pybot.runtime.hunt_policy import HuntPolicy
from pybot.runtime.hunt_tracks import HuntTracks, monotonic_ms
from pybot.runtime.input.input_backend import ShadowInputBackend
from pybot.runtime.logging import HuntLogger
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.validation_log import HuntValidationLogger
from pybot.runtime.detection.detector_session import DetectorSession
from pybot.runtime.workers.attack_loop import AttackLoop

FIXTURE = RECOGNITION_FIXTURES_DIR / "game-screenshots" / "333.png"


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
        discovery_interval_ms=3000,
        teleport_duration_ms=500,
        validation_enabled=False,
        control_file=None,
    )
    base.update(overrides)
    return HuntRuntimeConfig(**base)


def make_context(
    config: HuntRuntimeConfig,
    *,
    roi: HuntRoi,
    detector: DetectorSession,
    stop_event: threading.Event | None = None,
) -> HuntRuntimeContext:
    logger = HuntLogger(session_id="test_tracking_integration")
    tracks = HuntTracks()
    stop = stop_event or threading.Event()
    return HuntRuntimeContext(
        config=config,
        logger=logger,
        tracks=tracks,
        policy=HuntPolicy(),
        capture=FakeCapture(roi),
        detector=detector,
        tracker=detector,
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

    def test_discovery_creates_tracks(self) -> None:
        config = make_config()
        detector = FixtureDetector(self.roi_frame)
        ctx = make_context(config, roi=self.roi, detector=detector)

        # Discover from frame
        scan = detector.discover(self.roi)
        self.assertTrue(scan.ok)
        self.assertGreater(scan.accepted_count, 0)

        # Reconcile into tracks
        from pybot.recognition.rules import DiscoveryDetection

        detections = [
            DiscoveryDetection(x=d.x, y=d.y, confidence=d.confidence, candidate_scale=d.candidate_scale, living=True)
            for d in scan.detections
        ]
        summary = ctx.tracks.reconcile_detections(
            detections,
            mob_name="horn",
            now_tick=monotonic_ms(),
        )
        self.assertGreater(summary.added_count, 0)
        self.assertGreater(ctx.tracks.get_track_count(), 0)

    def test_shadow_attack_on_discovered_track(self) -> None:
        config = make_config(skill_delay_ms=0)
        detector = FixtureDetector(self.roi_frame)
        ctx = make_context(config, roi=self.roi, detector=detector)
        hunt_mode = create_hunt_mode(ctx, ShadowInputBackend())
        attack = AttackLoop(ctx, hunt_mode, ShadowInputBackend())

        # Discover from frame
        from pybot.recognition.rules import DiscoveryDetection

        scan = detector.discover(self.roi)
        detections = [
            DiscoveryDetection(x=d.x, y=d.y, confidence=d.confidence, candidate_scale=d.candidate_scale, living=True)
            for d in scan.detections
        ]
        ctx.tracks.reconcile_detections(detections, mob_name="horn", now_tick=monotonic_ms())

        now = monotonic_ms()
        target_id = ctx.policy.select_target(ctx.tracks.tracks_for_policy(now), now)
        self.assertIsNotNone(target_id)
        assert target_id is not None

        track_before = ctx.tracks.get_track_by_id(target_id)
        assert track_before is not None
        self.assertEqual(track_before.state, "alive")
        self.assertEqual(track_before.attack_count, 0)

        attack._attack_one(target_id, now)

        track_after = ctx.tracks.get_track_by_id(target_id)
        self.assertIsNotNone(track_after)
        if track_after is not None:
            self.assertEqual(track_after.state, "alive")
            self.assertEqual(track_after.attack_count, 1)

    def test_rediscovery_dedups_without_duplicates_or_moving(self) -> None:
        config = make_config()
        detector = FixtureDetector(self.roi_frame)
        ctx = make_context(config, roi=self.roi, detector=detector)

        from pybot.recognition.rules import DiscoveryDetection

        # Discover
        scan = detector.discover(self.roi)
        detections = [
            DiscoveryDetection(x=d.x, y=d.y, confidence=d.confidence, candidate_scale=d.candidate_scale, living=True)
            for d in scan.detections
        ]
        ctx.tracks.reconcile_detections(detections, mob_name="horn", now_tick=monotonic_ms())

        track = ctx.tracks.get_track_by_id(1)
        assert track is not None
        old_x, old_y = track.x, track.y
        count_before = ctx.tracks.get_track_count()

        # Re-discover the same mobs slightly shifted (within one object radius):
        # discovery is create-only, so it must NOT spawn duplicates and must NOT
        # move existing tracks (tracking owns position).
        detections2 = [
            DiscoveryDetection(x=d.x + 5, y=d.y + 5, confidence=d.confidence, candidate_scale=d.candidate_scale, living=True)
            for d in scan.detections
        ]
        summary = ctx.tracks.reconcile_detections(
            detections2, mob_name="horn", now_tick=monotonic_ms() + 1000
        )

        self.assertEqual(summary.added_count, 0)
        self.assertEqual(ctx.tracks.get_track_count(), count_before)
        self.assertEqual(track.x, old_x)
        self.assertEqual(track.y, old_y)

    def test_tracking_keeps_track_coords_fresh(self) -> None:
        config = make_config()
        detector = FixtureDetector(self.roi_frame)
        ctx = make_context(config, roi=self.roi, detector=detector)

        from pybot.runtime.detection.detector_session import StateTrackSnapshot

        scan = detector.discover(self.roi)
        detections = [
            DiscoveryDetection(x=d.x, y=d.y, confidence=d.confidence, candidate_scale=d.candidate_scale, living=True)
            for d in scan.detections
        ]
        ctx.tracks.reconcile_detections(detections, mob_name="horn", now_tick=monotonic_ms())

        snapshots = [
            StateTrackSnapshot(
                track_id=s.id,
                x=s.x,
                y=s.y,
                scale=s.discovery_scale if s.discovery_scale > 0 else 1.0,
            )
            for s in ctx.tracks.snapshot_alive(monotonic_ms())
        ]
        self.assertGreater(len(snapshots), 0)

        batch = detector.track_locals_frame(self.roi_frame, self.roi, snapshots)
        now = monotonic_ms() + 50
        dead_ids, lost_ids = ctx.tracks.apply_tracking(batch.results, now_tick=now)
        removed = dead_ids + lost_ids

        # Static fixture: at least one track is re-found and stays alive, and no
        # found track is dropped.
        self.assertGreater(batch.found_count, 0)
        found_ids = {r.track_id for r in batch.results if r.found}
        for track_id in found_ids:
            self.assertNotIn(track_id, removed)
            track = ctx.tracks.get_track_by_id(track_id)
            assert track is not None
            self.assertEqual(track.updated_tick, now)


if __name__ == "__main__":
    unittest.main()
