"""Tracking + discovery integration on fixture frames."""

from __future__ import annotations

import threading
import unittest

import cv2

from pybot.runtime.capture.window_roi import HuntRoi
from pybot.recognition.fixtures import default_horn_fixture
from pybot.recognition.rules import DiscoveryDetection
from pybot.config.runtime import HuntRuntimeConfig
from pybot.runtime.control import RuntimeControl
from pybot.runtime.hunt_mode import create_hunt_mode
from pybot.runtime.hunt_policy import HuntPolicy
from pybot.runtime.hunt_tracks import HuntTracks, monotonic_ms
from pybot.runtime.gate_controller import GateController
from pybot.runtime.input.input_backend import ShadowInputBackend
from pybot.runtime.logging import HuntLogger
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.validation_log import HuntValidationLogger
from pybot.runtime.detection.detector_session import DetectorSession
from pybot.runtime.workers.attack_loop import AttackLoop

from tests.runtime.fixtures import (
    FakeCapture,
    FixtureDetector,
    make_config,
    playfield_roi,
)

FIXTURE = default_horn_fixture()


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
        gates=GateController(),
    )
    ctx.stop_event = stop
    return ctx


class TrackingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        frame = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
        if frame is None:
            raise unittest.SkipTest("fixture Horn/3Horn.png missing")
        cls.roi_frame = playfield_roi(frame)
        cls.roi = HuntRoi(x=0, y=0, w=cls.roi_frame.shape[1], h=cls.roi_frame.shape[0])

    def test_discovery_publishes_candidates_then_tracking_creates_tracks(self) -> None:
        """Discovery publishes candidates; tracking creates tracks on fresh frame."""
        config = make_config()
        detector = FixtureDetector(self.roi_frame)
        ctx = make_context(config, roi=self.roi, detector=detector)

        # Discover from frame
        scan = detector.discover(self.roi)
        self.assertTrue(scan.ok)
        self.assertGreater(scan.accepted_count, 0)

        detections = [
            DiscoveryDetection(x=d.x, y=d.y, confidence=d.confidence, candidate_scale=d.candidate_scale, living=True)
            for d in scan.detections
        ]

        # Process discovery scan — matches/publishes candidates, does NOT create tracks
        summary = ctx.tracks.process_discovery_scan(
            detections,
            mob_name="horn",
            now_tick=monotonic_ms(),
        )
        self.assertGreater(summary.added_count, 0)
        # No tracks yet — tracking creates them
        self.assertEqual(ctx.tracks.get_track_count(), 0)

        # Tracking ingests candidates and creates tracks
        from pybot.runtime.detection.detector_session import StateTrackSnapshot

        candidates = ctx.tracks.get_and_clear_new_candidates()
        self.assertGreater(len(candidates), 0)

        for candidate in candidates:
            if candidate.candidate_scale <= 0:
                continue
            snap = StateTrackSnapshot(
                track_id=0,
                x=candidate.x,
                y=candidate.y,
                scale=candidate.candidate_scale,
            )
            batch = detector.track_locals_frame(self.roi_frame, self.roi, [snap])
            if batch.ok and batch.results and batch.results[0].found:
                r = batch.results[0]
                ctx.tracks.create_track(
                    "horn", r.x, r.y, candidate.confidence, candidate.candidate_scale,
                    now_tick=monotonic_ms(),
                )

        self.assertGreater(ctx.tracks.get_track_count(), 0)

    def test_shadow_attack_on_discovered_track(self) -> None:
        config = make_config(skill_delay_ms=0)
        detector = FixtureDetector(self.roi_frame)
        ctx = make_context(config, roi=self.roi, detector=detector)
        hunt_mode = create_hunt_mode(ctx, ShadowInputBackend())
        attack = AttackLoop(ctx, hunt_mode, ShadowInputBackend())

        # Create a track directly (simulating tracking post-candidate-ingest)
        scan = detector.discover(self.roi)
        detections = [
            DiscoveryDetection(x=d.x, y=d.y, confidence=d.confidence, candidate_scale=d.candidate_scale, living=True)
            for d in scan.detections
        ]
        # Process discovery to get candidates, then create tracks
        ctx.tracks.process_discovery_scan(detections, mob_name="horn", now_tick=monotonic_ms())
        candidates = ctx.tracks.get_and_clear_new_candidates()
        for candidate in candidates:
            ctx.tracks.create_track(
                "horn", candidate.x, candidate.y, candidate.confidence,
                candidate.candidate_scale, now_tick=monotonic_ms(),
            )

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

    def test_rediscovery_matches_without_duplicates_or_position_change(self) -> None:
        """Rediscovery matches existing tracks; tracking still owns position."""
        config = make_config()
        detector = FixtureDetector(self.roi_frame)
        ctx = make_context(config, roi=self.roi, detector=detector)

        # Create tracks directly
        scan = detector.discover(self.roi)
        detections = [
            DiscoveryDetection(x=d.x, y=d.y, confidence=d.confidence, candidate_scale=d.candidate_scale, living=True)
            for d in scan.detections
        ]
        ctx.tracks.process_discovery_scan(detections, mob_name="horn", now_tick=monotonic_ms())
        candidates = ctx.tracks.get_and_clear_new_candidates()
        for candidate in candidates:
            ctx.tracks.create_track(
                "horn", candidate.x, candidate.y, candidate.confidence,
                candidate.candidate_scale, now_tick=monotonic_ms(),
            )

        track = ctx.tracks.get_track_by_id(1)
        assert track is not None
        old_x, old_y = track.x, track.y
        count_before = ctx.tracks.get_track_count()

        # Re-discover the same mobs slightly shifted (within one object radius):
        # must NOT spawn candidates and must NOT move authoritative x/y
        # (tracking owns position).
        detections2 = [
            DiscoveryDetection(x=d.x + 5, y=d.y + 5, confidence=d.confidence, candidate_scale=d.candidate_scale, living=True)
            for d in scan.detections
        ]
        summary = ctx.tracks.process_discovery_scan(
            detections2, mob_name="horn", now_tick=monotonic_ms() + 1000
        )

        self.assertEqual(summary.added_count, 0)  # no new candidates
        self.assertEqual(ctx.tracks.get_track_count(), count_before)
        self.assertEqual(track.x, old_x)  # position unchanged
        self.assertEqual(track.y, old_y)
        self.assertEqual(track.discovery_miss_count, 0)  # miss count reset

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
        ctx.tracks.process_discovery_scan(detections, mob_name="horn", now_tick=monotonic_ms())
        candidates = ctx.tracks.get_and_clear_new_candidates()
        for candidate in candidates:
            ctx.tracks.create_track(
                "horn", candidate.x, candidate.y, candidate.confidence,
                candidate.candidate_scale, now_tick=monotonic_ms(),
            )

        snapshots = [
            StateTrackSnapshot(
                track_id=s.id,
                x=s.x,
                y=s.y,
                scale=s.discovery_scale,
            )
            for s in ctx.tracks.snapshot_alive(monotonic_ms())
            if s.discovery_scale > 0
        ]
        self.assertGreater(len(snapshots), 0)

        batch = detector.track_locals_frame(self.roi_frame, self.roi, snapshots)
        now = monotonic_ms() + 50
        missed_ids, _opacity_dead = ctx.tracks.apply_tracking(batch.results, now_tick=now)

        # Static fixture: at least one track is re-found and stays alive, and no
        # found track is dropped.
        self.assertGreater(batch.found_count, 0)
        found_ids = {r.track_id for r in batch.results if r.found}
        for track_id in found_ids:
            self.assertNotIn(track_id, missed_ids)
            track = ctx.tracks.get_track_by_id(track_id)
            assert track is not None
            self.assertEqual(track.updated_tick, now)


if __name__ == "__main__":
    unittest.main()
