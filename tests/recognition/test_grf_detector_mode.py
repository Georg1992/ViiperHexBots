"""GRF (modified sprite.grf) mode: colored-square discovery + tracking.

Modified sprites are one deterministic colored square. Distinct hunts get
distinct colors; paired map members share a color. GRF mode therefore:
- stores a palette+size marker descriptor (not animated silhouette refs);
- discovers with heatmap → blobs → square size/fill, not silhouette;
- tracks with palette fill of a descriptor-sized window.
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

    def test_modified_descriptor_is_marker_square(self) -> None:
        from pybot.mobs.marker_sprites import MARKER_SPRITE_SIZE, modified_sprite_rgb
        from pybot.recognition.detector.descriptors.descriptor_builder import (
            is_marker_square_descriptor,
        )

        ensure_mob_assets(log_fn=lambda _message: None)
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        descriptor = grf.ensure_descriptor("horn")
        self.assertTrue(is_marker_square_descriptor(descriptor))
        self.assertEqual(descriptor.avg_width, MARKER_SPRITE_SIZE)
        self.assertEqual(descriptor.avg_height, MARKER_SPRITE_SIZE)
        self.assertEqual(len(descriptor.match_palette_bgr), 1)
        self.assertFalse(descriptor.use_body_cluster_diversity)
        self.assertEqual(len(descriptor.silhouette_masks), 0)
        red, green, blue = modified_sprite_rgb("horn")
        self.assertEqual(
            tuple(descriptor.match_palette_bgr[0]),
            (blue, green, red),
        )

    def test_grf_discovery_rejects_wrong_marker_color(self) -> None:
        """A different-colored marker square must not be accepted as horn."""
        ensure_mob_assets(log_fn=lambda _message: None)
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        canvas = _modified_sprite_canvas("DesertWolf", "desert_wolf")
        impostor = grf.detect(canvas, "horn")
        self.assertEqual(
            len(impostor.accepted),
            0,
            "a different marker color must fail the GRF square gate",
        )
        self_match = grf.detect(canvas, "desert_wolf")
        self.assertGreater(
            len(self_match.accepted),
            0,
            "true modified sprite must still clear the square size/fill gate",
        )

    def test_grf_paired_members_share_marker_and_accept_each_other(self) -> None:
        """Isilla and Vanberk are the same orange square, so either hunt matches."""
        ensure_mob_assets(log_fn=lambda _message: None)
        from pybot.mobs.marker_sprites import modified_sprite_rgb

        self.assertEqual(modified_sprite_rgb("isilla"), modified_sprite_rgb("vanberk"))
        self.assertEqual(modified_sprite_rgb("merman"), modified_sprite_rgb("strouf"))
        self.assertNotEqual(modified_sprite_rgb("isilla"), modified_sprite_rgb("merman"))
        spr_path = MOBS_DIR / "isilla" / "modified_sprite" / "isilla.spr"
        if not spr_path.is_file():
            self.skipTest("isilla modified sprite is not installed")
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        canvas = _modified_sprite_canvas("isilla", "isilla")
        as_vanberk = grf.detect(canvas, "vanberk")
        self.assertGreater(
            len(as_vanberk.accepted),
            0,
            "shared-color pair member must match the other hunt descriptor",
        )

    def test_grf_patchy_breeze_self_matches(self) -> None:
        """Imported marker squares must still pass GRF size/fill."""
        ensure_mob_assets(log_fn=lambda _message: None)
        spr_path = MOBS_DIR / "breeze" / "modified_sprite" / "breeze.spr"
        if not spr_path.is_file():
            self.skipTest("breeze modified sprite is not installed")
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        canvas = _modified_sprite_canvas("breeze", "breeze")
        result = grf.detect(canvas, "breeze")
        self.assertGreater(
            len(result.accepted),
            0,
            "filled breeze square must clear the GRF size/fill gate",
        )

    def test_grf_strouf_self_matches(self) -> None:
        """Strouf marker square must clear size/fill on a live-sized ROI."""
        from pybot.recognition.detector.descriptors.descriptor_builder import (
            DescriptorBuilder,
        )

        spr_path = MOBS_DIR / "strouf" / "modified_sprite" / "strouf.spr"
        if not spr_path.is_file():
            self.skipTest("strouf modified sprite is not installed")
        DescriptorBuilder(ROOT).build_modified_sprite("strouf", force=True)
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        canvas = _modified_sprite_canvas("strouf", "strouf")
        frame = np.full((1024, 1024, 3), (30, 60, 30), dtype=np.uint8)
        y0 = (frame.shape[0] - canvas.shape[0]) // 2
        x0 = (frame.shape[1] - canvas.shape[1]) // 2
        frame[y0 : y0 + canvas.shape[0], x0 : x0 + canvas.shape[1]] = canvas
        result = grf.detect(frame, "strouf")
        self.assertGreater(
            len(result.accepted),
            0,
            "strouf modified sprite must clear the GRF size/fill gate",
        )

    def test_grf_evil_druid_body_only_self_matches(self) -> None:
        """Evil druid marker square must still pass size/fill."""
        from pybot.recognition.detector.descriptors.descriptor_builder import (
            DescriptorBuilder,
        )

        spr_path = MOBS_DIR / "evil_druid" / "modified_sprite" / "evil_druid.spr"
        if not spr_path.is_file():
            self.skipTest("evil_druid modified sprite is not installed")
        builder = DescriptorBuilder(ROOT)
        builder.build("evil_druid", force=True)
        builder.build_modified_sprite("evil_druid", force=True)
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        canvas = _modified_sprite_canvas("evil_druid", "evil_druid")
        result = grf.detect(canvas, "evil_druid")
        self.assertGreater(
            len(result.accepted),
            0,
            "evil druid marker square must clear the GRF size/fill gate",
        )

    def test_grf_discovery_skips_silhouette_gate(self) -> None:
        ensure_mob_assets(log_fn=lambda _message: None)
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        canvas = _modified_sprite_canvas("Horn", "horn")
        with patch.object(grf, "_evaluate_silhouette_gate") as silhouette:
            result = grf.detect(canvas, "horn")
        silhouette.assert_not_called()
        self.assertGreater(len(result.accepted), 0)

    def test_grf_score_at_skips_silhouette_gate(self) -> None:
        ensure_mob_assets(log_fn=lambda _message: None)
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        descriptor = grf.ensure_descriptor("horn")
        canvas = _modified_sprite_canvas("Horn", "horn")
        cx = canvas.shape[1] // 2
        cy = canvas.shape[0] // 2
        with patch.object(grf, "_evaluate_silhouette_gate") as silhouette:
            passed, _bbox, _fill = grf.score_at(canvas, descriptor, cx, cy)
        silhouette.assert_not_called()
        self.assertTrue(passed)

    def test_grf_marker_square_gate_rejects_skinny_blob(self) -> None:
        ensure_mob_assets(log_fn=lambda _message: None)
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        descriptor = grf.ensure_descriptor("horn")
        heatmap = np.ones((200, 200), dtype=np.float32)
        width = int(descriptor.avg_width)
        height = int(descriptor.avg_height)
        self.assertFalse(
            grf._passes_marker_square_gate((10, 10, 80, 20), descriptor, heatmap, 1.0),
        )
        self.assertTrue(
            grf._passes_marker_square_gate(
                (10, 10, width, height), descriptor, heatmap, 1.0,
            ),
        )

    def test_grf_heatmap_skips_edge_boost(self) -> None:
        ensure_mob_assets(log_fn=lambda _message: None)
        grf = MobDetector(ROOT, self.config, use_sprite_grf=True)
        canvas = _modified_sprite_canvas("Horn", "horn")
        with patch.object(
            grf.heatmap_detector,
            "_finish_heatmap",
            wraps=grf.heatmap_detector._finish_heatmap,
        ) as finish:
            grf.detect(canvas, "horn")
        self.assertFalse(finish.call_args.kwargs["edge_boost"])

    def test_modified_tracking_still_verifies_with_native_gate(self) -> None:
        """Static modified sprites still call score_at during local tracking."""
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
