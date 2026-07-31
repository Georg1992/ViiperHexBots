"""End-to-end shadow pipeline on fixture frame."""

from __future__ import annotations

import threading
import unittest

import cv2

from pybot.runtime.capture.window_roi import HuntRoi
from pybot.recognition.fixtures import default_horn_fixture
from pybot.runtime.control import RuntimeControl
from pybot.runtime.hunt_mode import create_hunt_mode
from pybot.runtime.hunt_policy import HuntPolicy
from pybot.runtime.hunt_tracks import HuntTracks
from pybot.runtime.gate_controller import GateController
from pybot.runtime.input.input_backend import ShadowInputBackend
from pybot.runtime.logging import HuntLogger
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.validation_log import HuntValidationLogger
from pybot.runtime.detection.detector_session import StateTrackSnapshot

from tests.runtime.fixtures import (
    FakeCapture,
    FixtureDetector,
    make_config,
    playfield_roi,
)

FIXTURE = default_horn_fixture()


class ShadowPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        frame = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
        if frame is None:
            raise unittest.SkipTest("fixture Horn/3Horn.png missing")
        cls.roi_frame = playfield_roi(frame)
        cls.roi = HuntRoi(x=0, y=0, w=cls.roi_frame.shape[1], h=cls.roi_frame.shape[0])

    def test_discovery_publishes_candidates_tracking_creates_tracks(self) -> None:
        config = make_config()
        logger = HuntLogger(session_id="test_shadow_pipeline")
        tracks = HuntTracks()
        stop = threading.Event()
        stop.set()
        capture = FakeCapture(self.roi)
        detector = FixtureDetector(self.roi_frame)
        ctx = HuntRuntimeContext(
            config=config,
            logger=logger,
            tracks=tracks,
            policy=HuntPolicy(),
            capture=capture,
            detector=detector,
            tracker=detector,
            validation=HuntValidationLogger(logger, tracks, enabled=False),
            control=RuntimeControl(None),
            gates=GateController(),
        )
        ctx.stop_event = stop
        hunt_mode = create_hunt_mode(ctx, ShadowInputBackend())

        from pybot.recognition.rules import DiscoveryDetection

        scan = detector.discover(self.roi)
        detections = [
            DiscoveryDetection(
                x=d.x, y=d.y, confidence=d.confidence, candidate_scale=d.candidate_scale, living=True
            )
            for d in scan.detections
        ]

        now = int(1_000_000)
        summary = tracks.process_discovery_scan(
            detections,
            mob_name="horn",
            now_tick=now,
        )

        self.assertGreater(summary.added_count, 0)
        # No tracks created yet — tracking owns that
        self.assertEqual(tracks.get_track_count(), 0)

        # Tracking ingests candidates and creates tracks on fresh frame
        candidates = tracks.get_and_clear_new_candidates()
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
                tracks.create_track(
                    "horn", r.x, r.y, candidate.confidence, candidate.candidate_scale,
                    now_tick=now,
                )

        self.assertGreater(tracks.get_track_count(), 0)
        track = tracks.get_track_by_id(1)
        assert track is not None
        self.assertGreater(track.updated_tick, 0)
        self.assertEqual(track.state, "alive")


if __name__ == "__main__":
    unittest.main()
