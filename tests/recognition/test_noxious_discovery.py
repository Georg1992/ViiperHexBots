"""Noxious discovery regressions for modified sprite.grf descriptors."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2

from pybot.paths import PROJECT_ROOT
from pybot.recognition.detector.detector import MobDetector, load_detector_config


class NoxiousDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = (
            PROJECT_ROOT
            / "pybot"
            / "recognition"
            / "test-fixtures"
            / "game-screenshots"
            / "Noxious"
            / "1Noxious.png"
        )
        cls.frame = cv2.imread(str(cls.fixture), cv2.IMREAD_COLOR)
        if cls.frame is None:
            raise unittest.SkipTest(f"missing Noxious fixture: {cls.fixture}")

    def test_sprite_grf_uses_fixed_four_x_work_scale(self) -> None:
        detector = MobDetector(
            PROJECT_ROOT,
            load_detector_config(),
            use_sprite_grf=True,
        )

        # GRF mode is deterministic: scale is selected from rendering mode, not
        # the selected mob descriptor or its dimensions.
        self.assertEqual(
            detector._discovery_heatmap_downscale(self.frame),
            4,
        )

    def test_sprite_grf_noxious_discovery_uses_fixed_four_x_heatmap(self) -> None:
        detector = MobDetector(
            PROJECT_ROOT,
            load_detector_config(),
            use_sprite_grf=True,
        )
        try:
            real_build = detector.heatmap_detector.build_sprite_heatmap
            with patch.object(
                detector.heatmap_detector,
                "build_sprite_heatmap",
                wraps=real_build,
            ) as build_heatmap:
                result = detector.detect(self.frame, "noxious")
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc)) from exc

        self.assertGreater(result.elapsed_s, 0.0)
        self.assertEqual(build_heatmap.call_args.kwargs["downscale"], 4)

    def test_normal_noxious_keeps_configured_scale_and_small_frames_fallback(self) -> None:
        config = load_detector_config()
        normal = MobDetector(PROJECT_ROOT, config, use_sprite_grf=False)
        grf = MobDetector(PROJECT_ROOT, config, use_sprite_grf=True)

        self.assertEqual(normal._discovery_heatmap_downscale(self.frame), 2)
        small_frame = self.frame[:700, :700]
        self.assertEqual(normal._discovery_heatmap_downscale(small_frame), 1)
        self.assertEqual(grf._discovery_heatmap_downscale(small_frame), 4)


if __name__ == "__main__":
    unittest.main()
