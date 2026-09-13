"""DetectorSession tests using fixture frames (no live capture)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2

from pybot.recognition.detector.tracking.local_tracker import LocalTrackResult

from pybot.paths import PROJECT_ROOT
from pybot.recognition.fixtures import default_horn_fixture
from pybot.runtime.capture.window_roi import HuntRoi
from pybot.runtime.detection.detector_session import DetectorSession, StateTrackSnapshot

ROOT = PROJECT_ROOT
FIXTURE = default_horn_fixture()


def playfield_roi(frame):
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


class DetectorSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
        if cls.frame is None:
            raise unittest.SkipTest("fixture Horn/3Horn.png missing")
        cls.roi_frame = playfield_roi(cls.frame)
        cls.roi = HuntRoi(x=0, y=0, w=cls.roi_frame.shape[1], h=cls.roi_frame.shape[0])
        cls.detector = DetectorSession("horn", project_root=ROOT)

    def test_discover_finds_living_candidates(self) -> None:
        scan = self.detector.discover_frame(self.roi_frame, self.roi)
        self.assertTrue(scan.ok)
        self.assertGreater(scan.raw_count, 0)
        living = [d for d in scan.detections if d.living]
        self.assertGreater(len(living), 0)
        self.assertGreater(scan.duration_ms, 0)
        # Stage timing splits into session-lock wait and detector work. An
        # uncontended call should spend almost all time in the detector itself.
        self.assertEqual(scan.duration_ms, scan.lock_wait_ms + scan.detect_ms)
        self.assertGreater(scan.detect_ms, 0)
        self.assertLess(scan.lock_wait_ms, 50)

    def test_discovery_heat_support_accepts_predicted_position(self) -> None:
        """A moving Track remains supported when heat is only at its prediction."""
        session = DetectorSession("horn", project_root=ROOT)
        try:
            heatmap = __import__("numpy").zeros((120, 120), dtype=float)
            heatmap[70, 80] = 1.0
            fake_result = type(
                "FakeDetectionResult",
                (),
                {
                    "sprite_heatmap": heatmap,
                    "candidates": [],
                    "accepted": [],
                    "elapsed_s": 0.001,
                    "timing": {},
                },
            )()
            with patch.object(session._detector, "detect", return_value=fake_result):
                scan = session.discover_frame(
                    heatmap,
                    HuntRoi(x=0, y=0, w=120, h=120),
                    heat_track_positions=[(7, 20, 20, 1.0, 80, 70)],
                )
            self.assertEqual(scan.heat_supported_track_ids, frozenset({7}))
        finally:
            session.close()

    def test_tracking_uses_one_shared_frame_in_snapshot_order(self) -> None:
        """All Tracks are updated sequentially from one immutable frame."""
        session = DetectorSession("horn", project_root=ROOT)
        snapshots = [
            StateTrackSnapshot(track_id=index + 1, x=100 + index * 40, y=100, scale=1.0)
            for index in range(7)
        ]
        calls: list[int] = []

        def fake_track(_frame, _mob_name, track, **_kwargs):
            calls.append(int(track["trackId"]))
            return LocalTrackResult(
                track_id=int(track["trackId"]), found=True,
                x=int(track["x"]), y=int(track["y"]),
                confidence=1.0, miss_reason="",
            )

        try:
            with patch.object(session._detector, "ensure_descriptor"):
                with patch.object(session._detector, "track_local", side_effect=fake_track):
                    batch = session.track_locals_frame(self.roi_frame, self.roi, snapshots)
        finally:
            session.close()

        self.assertEqual(calls, [snapshot.track_id for snapshot in snapshots])
        self.assertEqual([result.track_id for result in batch.results], calls)
        self.assertEqual(batch.found_count, len(snapshots))
        self.assertEqual(set(batch.track_durations_ms), set(calls))

    def test_reset_clears_visual_state_without_async_completion_queue(self) -> None:
        """Reset clears state; the single tracking path has no late callbacks."""
        session = DetectorSession("horn", project_root=ROOT)
        session.clear_track_states()
        self.assertFalse(hasattr(session, "submit_track_locals_frame"))
        self.assertEqual(getattr(session._detector, "_local_track_states", {}), {})
        session.close()

    def test_clear_states_leaves_visual_state_empty(self) -> None:
        """An area reset cannot leave old visual state behind."""
        session = DetectorSession("horn", project_root=ROOT)
        session.clear_track_states()
        self.assertEqual(getattr(session._detector, "_local_track_states", {}), {})
        session.close()

    def test_track_locals_returns_results(self) -> None:
        scan = self.detector.discover_frame(self.roi_frame, self.roi)
        anchor = next(d for d in scan.detections if d.living)
        snapshots = [
            StateTrackSnapshot(
                track_id=1,
                x=anchor.x,
                y=anchor.y,
                scale=anchor.candidate_scale,
            )
        ]
        batch = self.detector.track_locals_frame(self.roi_frame, self.roi, snapshots)
        self.assertTrue(batch.ok)
        self.assertEqual(len(batch.results), 1)
        self.assertTrue(batch.results[0].found)
        self.assertGreater(batch.duration_ms, 0)
        # Stage timing splits into session-lock wait and local-follow compute.
        self.assertEqual(batch.duration_ms, batch.lock_wait_ms + batch.compute_ms)
        self.assertGreater(batch.compute_ms, 0)
        self.assertLess(batch.lock_wait_ms, 50)

    def test_isilla_vanberk_merges_overlapping_similar_sprites(self) -> None:
        """Both descriptors fire on one blob; keep the stronger hit."""
        session = DetectorSession("isilla+vanberk", project_root=ROOT)
        heatmap = __import__("numpy").zeros((80, 80), dtype=float)
        heatmap[50, 50] = 1.0

        def fake_detect(_frame, mob_name, **_kwargs):
            score = 0.91 if mob_name == "isilla" else 0.72
            hit = type(
                "Candidate",
                (),
                {
                    "center_x": 50 if mob_name == "isilla" else 52,
                    "center_y": 50,
                    "final_score": score,
                    "candidate_scale": 1.0,
                    "accepted": True,
                    "bbox": (40, 40, 20, 20),
                },
            )()
            return type(
                "FakeDetectionResult",
                (),
                {
                    "sprite_heatmap": heatmap,
                    "candidates": [hit],
                    "accepted": [hit],
                    "elapsed_s": 0.001,
                    "timing": {},
                },
            )()

        try:
            with patch.object(session._detector, "detect", side_effect=fake_detect):
                scan = session.discover_frame(
                    heatmap,
                    HuntRoi(x=0, y=0, w=80, h=80),
                )
            self.assertEqual(scan.accepted_count, 1)
            self.assertEqual(scan.detections[0].mob_name, "isilla")
            self.assertEqual(scan.raw_count, 2)
        finally:
            session.close()

    def test_isilla_vanberk_keeps_separated_mobs(self) -> None:
        session = DetectorSession("isilla+vanberk", project_root=ROOT)
        heatmap = __import__("numpy").zeros((80, 80), dtype=float)

        def fake_detect(_frame, mob_name, **_kwargs):
            x = 10 if mob_name == "isilla" else 70
            hit = type(
                "Candidate",
                (),
                {
                    "center_x": x,
                    "center_y": 40,
                    "final_score": 0.8,
                    "candidate_scale": 1.0,
                    "accepted": True,
                    "bbox": (x - 10, 30, 20, 20),
                },
            )()
            return type(
                "FakeDetectionResult",
                (),
                {
                    "sprite_heatmap": heatmap,
                    "candidates": [hit],
                    "accepted": [hit],
                    "elapsed_s": 0.001,
                    "timing": {},
                },
            )()

        try:
            with patch.object(session._detector, "detect", side_effect=fake_detect):
                scan = session.discover_frame(
                    heatmap,
                    HuntRoi(x=0, y=0, w=80, h=80),
                )
            names = sorted(item.mob_name for item in scan.detections)
            self.assertEqual(names, ["isilla", "vanberk"])
        finally:
            session.close()

    def test_isilla_vanberk_heat_from_either_palette_supports_track(self) -> None:
        """Shared palette: Vanberk heat keeps an Isilla-acquired Track alive."""
        session = DetectorSession("isilla+vanberk", project_root=ROOT)
        empty = __import__("numpy").zeros((80, 80), dtype=float)
        vanberk_heat = empty.copy()
        vanberk_heat[40, 20] = 1.0

        def fake_detect(_frame, mob_name, **_kwargs):
            heatmap = vanberk_heat if mob_name == "vanberk" else empty
            return type(
                "FakeDetectionResult",
                (),
                {
                    "sprite_heatmap": heatmap,
                    "candidates": [],
                    "accepted": [],
                    "elapsed_s": 0.001,
                    "timing": {},
                },
            )()

        try:
            with patch.object(session._detector, "detect", side_effect=fake_detect):
                scan = session.discover_frame(
                    empty,
                    HuntRoi(x=0, y=0, w=80, h=80),
                    heat_track_positions=[(3, 20, 40, 1.0)],
                )
            self.assertEqual(scan.heat_supported_track_ids, frozenset({3}))
        finally:
            session.close()

    def test_isilla_vanberk_tracking_uses_snapshot_sprite(self) -> None:
        session = DetectorSession("isilla+vanberk", project_root=ROOT)
        seen: list[str] = []

        def fake_track(_frame, mob_name, track, **_kwargs):
            seen.append(mob_name)
            return LocalTrackResult(
                track_id=int(track["trackId"]),
                found=True,
                x=int(track["x"]),
                y=int(track["y"]),
                confidence=1.0,
                miss_reason="",
            )

        try:
            with patch.object(session._detector, "ensure_descriptor"):
                with patch.object(session._detector, "track_local", side_effect=fake_track):
                    batch = session.track_locals_frame(
                        self.roi_frame,
                        self.roi,
                        [
                            StateTrackSnapshot(
                                track_id=1, x=10, y=10, scale=1.0, mob_name="vanberk"
                            ),
                            StateTrackSnapshot(
                                track_id=2, x=40, y=10, scale=1.0, mob_name="isilla"
                            ),
                        ],
                    )
            self.assertEqual(seen, ["vanberk", "isilla"])
            self.assertEqual(batch.found_count, 2)
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
