"""GRF (modified sprite.grf) mode: strict gate + static single-frame tracking.

Modified sprites are one deterministic static frame with a distinctive red
palette. GRF mode therefore:
- references that single frame (the modified descriptor carries one unique
  silhouette pose — ``descriptor_is_static``);
- uses the same strict silhouette recall and precision thresholds as normal
  animated sprites;
- widens the extract aspect band (``grfAspectBandScale``) so a clipped
  palette CC (e.g. Anubis head shade outside the match radius) is not
  rejected before the silhouette match;
- keeps native-resolution silhouette verification during local tracking.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.fixtures import default_horn_fixture
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.detector.tracking import local_tracker
from pybot.recognition.detector.tracking.local_tracker import track_local

ROOT = PROJECT_ROOT


def playfield_roi(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


class GrfDetectorModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_detector_config()


    def test_silhouette_gate_thresholds_are_strict_in_both_modes(self) -> None:
        normal = MobDetector(ROOT, self.config)
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        normal_thresholds = normal.silhouette_gate_thresholds()
        grf_thresholds = grf.silhouette_gate_thresholds()
        expected = (
            float(self.config["minSilhouetteRecall"]),
            float(self.config["minSilhouettePrecision"]),
        )
        self.assertEqual(normal_thresholds, expected)
        self.assertEqual(grf_thresholds, expected)

    def test_modified_descriptors_are_static_single_frame(self) -> None:
        """The modified descriptor references the one static frame (all refs identical)."""
        from pybot.mobs.catalog import ensure_mob_assets

        ensure_mob_assets(log_fn=lambda _message: None)
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
        from pybot.mobs.catalog import ensure_mob_assets

        ensure_mob_assets(log_fn=lambda _message: None)
        normal = MobDetector(ROOT, self.config)
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        descriptor = grf.ensure_descriptor("anubis")
        n_min, n_max = normal._effective_aspect_band(descriptor)
        g_min, g_max = grf._effective_aspect_band(descriptor)
        self.assertEqual((n_min, n_max), (descriptor.min_aspect_ratio, descriptor.max_aspect_ratio))
        self.assertLess(g_min, n_min)
        self.assertGreater(g_max, n_max)

    def test_modified_tracking_still_verifies_with_native_gate(self) -> None:
        """Static modified sprites cannot bypass the strict silhouette gate."""
        detector = MobDetector(ROOT, load_detector_config(), use_sprite_grf=True)
        descriptor = detector.ensure_descriptor("horn")
        frame = np.zeros((400, 400, 3), dtype=np.uint8)

        def fake_local_heatmap(_heatmap_detector, work_bgr, _descriptor, _scale):
            heatmap = np.zeros(work_bgr.shape[:2], dtype=np.float32)
            heatmap[heatmap.shape[0] // 2, heatmap.shape[1] // 2] = 1.0
            return heatmap

        with (
            patch.object(local_tracker, "_build_local_follow_heatmap", side_effect=fake_local_heatmap),
            patch.object(
                detector,
                "score_at",
                return_value=(True, (190, 190, 20, 20), 0.9),
            ) as score_at,
        ):
            result = local_tracker._find_local_peak(
                detector,
                frame,
                descriptor,
                200,
                200,
                1.0,
                search_radius_px=20,
            )

        self.assertIsNotNone(result)
        score_at.assert_called_once()

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


if __name__ == "__main__":
    unittest.main()
