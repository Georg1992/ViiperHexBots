"""GRF (modified sprite.grf) mode: relaxed gate + static single-frame tracking.

Modified sprites are one deterministic static frame with a distinctive red
palette. GRF mode therefore:
- references that single frame (the modified descriptor carries one unique
  silhouette pose — ``descriptor_is_static``);
- relaxes the silhouette gate (``grfMinSilhouetteRecall/Precision``);
- widens the extract aspect band (``grfAspectBandScale``) so a clipped
  palette CC (e.g. Anubis head shade outside the match radius) is not
  rejected before the silhouette match;
- lets local tracking follow a peak without the expensive native-resolution
  silhouette verify (``grfLocalTrackSkipNativeGate``).
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from pybot.paths import PROJECT_ROOT, RECOGNITION_DIR
from pybot.recognition.fixtures import (
    MOB_FIXTURE_SUITES,
    default_horn_fixture,
    fixture_search_frame,
)
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.detector.tracking.local_tracker import (
    _refine_hit_to_sprite_center,
    track_local,
)

ROOT = PROJECT_ROOT
MOB_REC = RECOGNITION_DIR


def playfield_roi(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


def _anubis_modified_frame() -> np.ndarray:
    suite = next(s for s in MOB_FIXTURE_SUITES if s.mob_name == "anubis")
    image = next(
        image for image in suite.images() if "ModifiedSprite" in image.file_name
    )
    frame = cv2.imread(str(image.path), cv2.IMREAD_COLOR)
    assert frame is not None, "anubis ModifiedSprite fixture missing"
    return fixture_search_frame(frame)


class GrfDetectorModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_detector_config()

    def test_silhouette_gate_thresholds_relaxed_in_grf_mode(self) -> None:
        normal = MobDetector(ROOT, self.config)
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        n_recall, n_precision = normal.silhouette_gate_thresholds()
        g_recall, g_precision = grf.silhouette_gate_thresholds()
        self.assertLess(g_recall, n_recall)
        self.assertLess(g_precision, n_precision)
        self.assertEqual(n_recall, float(self.config["minSilhouetteRecall"]))
        self.assertEqual(n_precision, float(self.config["minSilhouettePrecision"]))
        self.assertEqual(g_recall, float(self.config["grfMinSilhouetteRecall"]))
        self.assertEqual(g_precision, float(self.config["grfMinSilhouettePrecision"]))

    def test_modified_descriptors_are_static_single_frame(self) -> None:
        """The modified descriptor references the one static frame (all refs identical)."""
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        for mob in ("anubis", "horn"):
            descriptor = grf.ensure_descriptor(mob)
            self.assertTrue(
                grf.descriptor_is_static(descriptor),
                f"modified descriptor for {mob} should be single-frame",
            )

    def test_normal_descriptor_is_not_static(self) -> None:
        """Animated originals keep pose diversity and full gate verification."""
        normal = MobDetector(ROOT, self.config)
        horn = normal.ensure_descriptor("horn")
        self.assertFalse(normal.descriptor_is_static(horn))

    def test_grf_aspect_band_widened(self) -> None:
        normal = MobDetector(ROOT, self.config)
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        descriptor = grf.ensure_descriptor("anubis")
        n_min, n_max = normal._effective_aspect_band(descriptor)
        g_min, g_max = grf._effective_aspect_band(descriptor)
        self.assertEqual((n_min, n_max), (descriptor.min_aspect_ratio, descriptor.max_aspect_ratio))
        self.assertLess(g_min, n_min)
        self.assertGreater(g_max, n_max)

    def test_anubis_modified_fixture_detects_in_grf_mode(self) -> None:
        """Regression: the clipped Anubis extract was rejected by the tight aspect
        band even on a clean fixture, so the bot missed tracked mobs and
        teleported away. GRF mode must detect it."""
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        result = grf.detect(_anubis_modified_frame(), "anubis")
        self.assertGreater(len(result.accepted), 0)
        passed = [c for c in result.silhouette_checks if c.passed]
        self.assertGreater(len(passed), 0)
        self.assertGreater(passed[0].recall, 0.0)
        self.assertGreater(passed[0].precision, 0.0)

    def test_grf_static_tracking_skips_native_gate(self) -> None:
        """Static GRF descriptors follow a peak without the native verify — the
        dominant per-tick cost for large Anubis sprites."""
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        frame = _anubis_modified_frame()
        discovery = grf.detect(frame, "anubis")
        self.assertGreater(len(discovery.accepted), 0)
        anchor = discovery.accepted[0]
        track = {
            "trackId": -1,
            "x": anchor.center_x,
            "y": anchor.center_y,
            "scale": anchor.candidate_scale,
        }
        with patch.object(
            grf,
            "score_at",
            side_effect=AssertionError("native gate must be skipped in GRF static mode"),
        ):
            result = track_local(grf, frame, "anubis", track)
        self.assertTrue(result.found, result.miss_reason)

    def test_normal_tracking_still_verifies_with_native_gate(self) -> None:
        """Animated originals keep the full silhouette verify on every reacquire."""
        config = load_detector_config()
        detector = MobDetector(ROOT, config)
        frame = cv2.imread(str(default_horn_fixture()), cv2.IMREAD_COLOR)
        assert frame is not None, "Horn fixture missing"
        roi = playfield_roi(frame)
        discovery = detector.detect(roi, "horn")
        living = [c for c in discovery.accepted]
        self.assertGreater(len(living), 0)
        track = {
            "trackId": -2,
            "x": living[0].center_x,
            "y": living[0].center_y,
            "scale": living[0].candidate_scale,
        }
        with patch.object(detector, "score_at", wraps=detector.score_at) as spy:
            result = track_local(detector, roi, "horn", track)
        self.assertTrue(result.found, result.miss_reason)
        spy.assert_called()

    def test_refine_hit_to_sprite_center_returns_sprite_bbox_center(self) -> None:
        """The aim point snaps to the palette-CC bbox center, not the raw input."""
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        descriptor = grf.ensure_descriptor("anubis")
        frame = np.zeros((800, 800, 3), dtype=np.uint8)
        # A solid red body at a known location, comfortably inside the
        # descriptor-sized window (BGR red matches the palette).
        frame[320:440, 320:440] = (0, 0, 220)
        # Input deliberately off-center — like a heat peak on the dense region.
        cx, cy = _refine_hit_to_sprite_center(
            grf, frame, descriptor, 395, 370, 1.0,
        )
        self.assertLess(abs(cx - 380), 4)  # bbox center x = (320+440)/2
        self.assertLess(abs(cy - 380), 4)  # bbox center y = (320+440)/2

    def test_fast_tracking_sticks_to_sprite_center(self) -> None:
        """A deliberately off-sprite seed still resolves onto the mob's center."""
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        frame = _anubis_modified_frame()
        discovery = grf.detect(frame, "anubis")
        self.assertGreater(len(discovery.accepted), 0)
        anchor = discovery.accepted[0]
        track = {
            "trackId": -7,
            "x": anchor.center_x + 45,
            "y": anchor.center_y - 60,
            "scale": anchor.candidate_scale,
        }
        result = track_local(grf, frame, "anubis", track)
        self.assertTrue(result.found, result.miss_reason)
        self.assertLess(abs(result.x - anchor.center_x), 20)
        self.assertLess(abs(result.y - anchor.center_y), 20)


if __name__ == "__main__":
    unittest.main()
