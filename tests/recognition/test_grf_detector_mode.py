"""GRF (modified sprite.grf) mode: strict gate + static single-frame tracking.

Modified sprites are one deterministic static frame with a distinctive red
palette. GRF mode therefore:
- references that single frame (the modified descriptor carries one unique
  silhouette pose — ``descriptor_is_static``);
- uses stricter silhouette recall and precision floors than animated sprites,
  because every modified mob shares the same red palette and shape is the
  only discriminator (alligator must not pass as frilldora);
- skips noisy-candidate silhouette deform so a same-color impostor crop
  cannot be warped toward the hunt reference;
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

from pybot.mobs.catalog import MOBS_DIR, ensure_mob_assets
from pybot.paths import PROJECT_ROOT
from pybot.recognition.act_reader import ActReader
from pybot.recognition.fixtures import default_horn_fixture
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.detector.tracking import local_tracker
from pybot.recognition.detector.tracking.local_tracker import track_local
from pybot.recognition.frame_renderer import render_act_frame
from pybot.recognition.spr_reader import SprReader

ROOT = PROJECT_ROOT


def _modified_sprite_canvas(asset_folder: str, spr_stem: str) -> np.ndarray:
    spr_path = MOBS_DIR / asset_folder / "modified_sprite" / f"{spr_stem}.spr"
    act_path = MOBS_DIR / asset_folder / "modified_sprite" / f"{spr_stem}.act"
    if not spr_path.is_file() or not act_path.is_file():
        raise unittest.SkipTest(f"missing modified sprite pair: {spr_path}")
    spr = SprReader(spr_path).load()
    act = ActReader(act_path).load()
    bgra = render_act_frame(spr, act.actions[0].frames[0])
    ys, xs = np.where(bgra[:, :, 3] > 0)
    if len(xs) == 0:
        raise unittest.SkipTest(f"modified sprite {spr_stem} has no opaque pixels")
    crop = bgra[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
    height, width = crop.shape[:2]
    pad = 80
    canvas = np.full((height + 2 * pad, width + 2 * pad, 3), (30, 60, 30), dtype=np.uint8)
    alpha = crop[:, :, 3] > 0
    canvas[pad : pad + height, pad : pad + width][alpha] = crop[:, :, :3][alpha]
    return canvas


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

    def test_silhouette_gate_thresholds_are_stricter_in_grf_mode(self) -> None:
        normal = MobDetector(ROOT, self.config)
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        normal_recall, normal_precision = normal.silhouette_gate_thresholds()
        grf_recall, grf_precision = grf.silhouette_gate_thresholds()
        self.assertEqual(
            (normal_recall, normal_precision),
            (
                float(self.config["minSilhouetteRecall"]),
                float(self.config["minSilhouettePrecision"]),
            ),
        )
        self.assertEqual(
            (grf_recall, grf_precision),
            (
                float(self.config["grfMinSilhouetteRecall"]),
                float(self.config["grfMinSilhouettePrecision"]),
            ),
        )
        self.assertGreater(grf_recall, normal_recall)
        self.assertGreater(grf_precision, normal_precision)

    def test_grf_discovery_rejects_same_color_wrong_shape(self) -> None:
        """A red desert wolf must not be accepted while hunting horn."""
        ensure_mob_assets(log_fn=lambda _message: None)
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        canvas = _modified_sprite_canvas("DesertWolf", "desert_wolf")
        impostor = grf.detect(canvas, "horn")
        self.assertEqual(
            len(impostor.accepted),
            0,
            "same-color modified sprite of a different mob must fail the GRF gate",
        )
        self_match = grf.detect(canvas, "desert_wolf")
        self.assertGreater(
            len(self_match.accepted),
            0,
            "true modified sprite must still clear the stricter GRF floors",
        )

    def test_grf_skips_noisy_candidate_deform(self) -> None:
        detector = MobDetector(ROOT, self.config, use_sprite_grf=True)
        candidate = np.ones((16, 16), dtype=np.float32)
        with patch.object(detector, "_deform_silhouette_occupancy") as deform:
            result = detector._maybe_deform_noisy_candidate(
                candidate,
                [],
                np.zeros((8, 8, 3), dtype=np.uint8),
                None,
                np.zeros((1, 3), dtype=np.float32),
                10.0,
                None,
            )
        deform.assert_not_called()
        self.assertIs(result, candidate)

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
