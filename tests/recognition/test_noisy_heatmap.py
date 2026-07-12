"""Tests for gated noisy-heatmap background correction."""

from __future__ import annotations

import unittest

import cv2

from pybot.paths import PROJECT_ROOT
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.detector.scoring.noise_analyzer import analyze_heatmap_noise
from pybot.recognition.fixtures import MOB_FIXTURE_SUITES, fixture_search_frame
from pybot.mobs.catalog import descriptor_path
from pybot.recognition.detector.descriptors.descriptor import MobDescriptor


class NoisyHeatmapGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_detector_config()
        cls.detector = MobDetector(PROJECT_ROOT, cls.config)

    def test_noise_gate_only_triggers_normal_alligator(self) -> None:
        expected_noisy = {
            ("Alligator", "1Alligator.png"),
            ("Alligator", "2Alligator.png"),
            ("Alligator", "3Alligator.png"),
            ("Alligator", "4Alligator.png"),
        }
        for suite in MOB_FIXTURE_SUITES:
            descriptor = MobDescriptor.load(descriptor_path(suite.mob_name))
            for image in suite.images():
                frame = fixture_search_frame(cv2.imread(str(image.path), cv2.IMREAD_COLOR))
                noise = analyze_heatmap_noise(
                    frame,
                    descriptor,
                    max_sprite_palette_distance=float(self.config["maxSpritePaletteDistance"]),
                    hot_frac_min=float(self.config["noisyHeatmapHotFracMin"]),
                    raw_heat_threshold=float(self.config["noisyHeatmapRawHeatThreshold"]),
                )
                key = (suite.folder, image.file_name)
                with self.subTest(fixture=key):
                    if key in expected_noisy:
                        self.assertTrue(noise.is_noisy, f"expected noisy heatmap for {key}")
                    else:
                        self.assertFalse(noise.is_noisy, f"expected clean heatmap for {key}")

    def test_background_correction_applied_only_when_noisy(self) -> None:
        for suite in MOB_FIXTURE_SUITES:
            for image in suite.images():
                frame = fixture_search_frame(cv2.imread(str(image.path), cv2.IMREAD_COLOR))
                result = self.detector.detect(frame, suite.mob_name)
                key = (suite.folder, image.file_name)
                with self.subTest(fixture=key):
                    self.assertEqual(
                        result.background_corrected,
                        result.noisy_heatmap,
                        f"background_corrected must mirror noisy gate for {key}",
                    )
                    if suite.mob_name == "alligator" and not image.gray_world:
                        self.assertTrue(result.background_corrected, key)
                    else:
                        self.assertFalse(result.background_corrected, key)


if __name__ == "__main__":
    unittest.main()
