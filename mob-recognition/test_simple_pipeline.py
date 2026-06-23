from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
MOB_REC = Path(__file__).resolve().parent
SIMPLE = MOB_REC / "simple"
for path in (str(MOB_REC), str(SIMPLE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cli import apply_scale_calibration, parse_scale_range  # noqa: E402
from dataset_runner import run_fixtures  # noqa: E402
from descriptor_builder import SimpleDescriptorBuilder  # noqa: E402
from detector import SimpleMobDetector, load_simple_config  # noqa: E402
from death_validator import DeathValidator  # noqa: E402


class SimplePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_simple_config()
        cls.descriptor = SimpleDescriptorBuilder(ROOT).build("horn")
        cls.detector = SimpleMobDetector(ROOT, cls.config)
        cls.fixture_dir = MOB_REC / "test-fixtures" / "game-screenshots"

    def _image(self, name: str):
        image = cv2.imread(str(self.fixture_dir / name), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image, name)
        return image

    def test_descriptor_builds_from_spr_act(self) -> None:
        self.assertGreater(self.descriptor.template_count, 0)
        self.assertGreater(self.descriptor.avg_width, 10)
        self.assertGreater(self.descriptor.avg_height, 10)
        self.assertGreater(len(self.descriptor.body_colors), 0)
        self.assertGreater(len(self.descriptor.accent_colors), 0)
        self.assertIsNotNone(self.descriptor.dead)
        self.assertGreater(self.descriptor.dead.size.avg_width, self.descriptor.size.avg_width)

    def test_opacity_fade_detects_blended_sprite(self) -> None:
        import numpy as np

        validator = DeathValidator(self.config, self.detector.region_scorer)
        background = np.array([128.0, 128.0, 128.0], dtype=np.float32)
        foreground = np.array([40.0, 30.0, 90.0], dtype=np.float32)
        palette = [(40, 30, 90)]

        full_pixels = np.tile(foreground, (100, 1))
        full_opacity = validator._estimate_mean_opacity(full_pixels, background, palette)
        self.assertGreater(full_opacity, 0.95)

        faded_pixels = np.tile(foreground * 0.55 + background * 0.45, (100, 1))
        faded_opacity = validator._estimate_mean_opacity(faded_pixels, background, palette)
        self.assertLess(faded_opacity, 0.90)
        self.assertLess(faded_opacity, full_opacity)

    def test_heatmaps_are_generated(self) -> None:
        result = self.detector.detect(self._image("333.png"), "horn")
        self.assertEqual(result.heatmaps.body_palette.shape, result.heatmaps.final_center.shape)
        self.assertGreater(float(result.heatmaps.final_center.max()), 0.0)

    def test_candidate_centers_are_produced(self) -> None:
        result = self.detector.detect(self._image("333.png"), "horn")
        self.assertGreater(len(result.candidates), 0)

    def test_region_scorer_returns_breakdown(self) -> None:
        result = self.detector.detect(self._image("333.png"), "horn")
        candidate = result.candidates[0]
        self.assertGreaterEqual(candidate.final_score, 0.0)
        self.assertGreaterEqual(candidate.body_palette_score, 0.0)
        self.assertGreaterEqual(candidate.accent_score, 0.0)
        self.assertGreaterEqual(candidate.rare_color_score, 0.0)
        self.assertGreaterEqual(candidate.local_pattern_score, 0.0)
        self.assertGreaterEqual(candidate.color_purity_score, 0.0)
        self.assertGreaterEqual(candidate.size_score, 0.0)
        self.assertGreater(candidate.candidate_scale, 0.0)

    def test_scale_calibration_config_overrides_scales(self) -> None:
        scale_range = parse_scale_range("0.45,0.55")
        calibrated = apply_scale_calibration(self.config, scale_range, enforce_size_gate=True)
        self.assertEqual(calibrated["scales"], [0.45, 0.5, 0.55])
        self.assertEqual(calibrated["centerScales"], [0.45, 0.5, 0.55])
        self.assertTrue(calibrated["enforceObjectSizeGate"])

    def test_clear_negative_has_no_accepts(self) -> None:
        result = self.detector.detect(self._image("qqq.png"), "horn")
        self.assertEqual(len(result.accepted), 0)

    def test_fixtures_simple_runs_end_to_end(self) -> None:
        summary = run_fixtures(
            "horn",
            MOB_REC / "test-fixtures",
            debug=False,
            rebuild_descriptor=False,
        )
        self.assertGreater(len(summary["images"]), 0)
        self.assertIn("tp", summary["totals"])


if __name__ == "__main__":
    unittest.main()
