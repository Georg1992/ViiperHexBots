"""Color-structure pre-silhouette gate (groups, mono-family, foreign palette)."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.detector.scoring.heatmap_detector import (
    required_groups_structure,
)
from pybot.recognition.fixtures import fixture_search_frame


class ColorStructureGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.detector = MobDetector(PROJECT_ROOT, load_detector_config())
        cls.descriptor = cls.detector.ensure_descriptor("wild_rose")
        path = (
            PROJECT_ROOT
            / "pybot"
            / "recognition"
            / "test-fixtures"
            / "game-screenshots"
            / "WildRose"
            / "2WildRose_Gray.png"
        )
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert frame is not None
        cls.frame = fixture_search_frame(frame)

    def _heat_crop(
        self,
        frame: np.ndarray,
        descriptor,
        center_xy: tuple[float, float],
    ) -> tuple[tuple[int, int, int, int], np.ndarray]:
        downscale = 1
        if (
            self.detector.discovery_heatmap_downscale > 1
            and min(frame.shape[0], frame.shape[1])
            >= self.detector.discovery_heatmap_downscale_min_side
        ):
            downscale = self.detector.discovery_heatmap_downscale
        heatmap = self.detector.heatmap_detector.build_sprite_heatmap(
            frame, descriptor, downscale=downscale
        )
        blobs = self.detector.heatmap_detector.top_centers(heatmap, descriptor)
        tx, ty = center_xy
        for cx, cy, _heat, comp in blobs:
            if abs(cx - tx) < 20 and abs(cy - ty) < 20:
                bx, by, bw, bh = comp
                crop = frame[by : by + bh, bx : bx + bw]
                return comp, crop
        self.fail(f"no heat blob near {center_xy}")
        raise AssertionError("unreachable")

    def test_wild_rose_crop_passes_structure(self) -> None:
        comp, crop = self._heat_crop(self.frame, self.descriptor, (163, 262))
        present, second, coverage, body_strong = required_groups_structure(
            crop,
            self.descriptor,
            float(self.descriptor.max_sprite_palette_distance),
        )
        self.assertGreaterEqual(present, 2)
        self.assertGreaterEqual(second, 0.09)
        self.assertGreaterEqual(coverage, 0.28)
        self.assertGreaterEqual(body_strong, 0.03)
        self.assertTrue(
            self.detector._passes_color_structure_gate(
                self.frame, self.descriptor, comp
            )
        )

    def test_zero_wild_rose_gray_rejects_impostor(self) -> None:
        """0WildRose_Gray: tiny heat CC must not clear body_strong via cache/inflation."""
        path = (
            PROJECT_ROOT
            / "pybot"
            / "recognition"
            / "test-fixtures"
            / "game-screenshots"
            / "WildRose"
            / "0WildRose_Gray.png"
        )
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(frame)
        frame = fixture_search_frame(frame)
        result = self.detector.detect(frame, "wild_rose")
        self.assertEqual(len(result.accepted), 0)


    def test_poring_crop_fails_body_strong(self) -> None:
        # Locate Poring on the base palette heatmap (body diversity may press
        # the final heat peak away). Foreign pink blob has no Wild Rose body.
        from pybot.recognition.detector.scoring.heatmap_detector import (
            weighted_sprite_palette_heatmap,
        )

        base, _sim = weighted_sprite_palette_heatmap(
            self.frame,
            self.descriptor,
            float(self.descriptor.max_sprite_palette_distance),
            return_similarity=True,
        )
        y0, x0 = 400, 800
        patch = base[y0 : y0 + 80, x0 : x0 + 80]
        py, px = np.unravel_index(int(patch.argmax()), patch.shape)
        cx, cy = int(x0 + px), int(y0 + py)
        size = 24
        comp = (cx - size // 2, cy - size // 2, size, size)
        x, y, bw, bh = comp
        crop = self.frame[y : y + bh, x : x + bw]
        _present, _second, _coverage, body_strong = required_groups_structure(
            crop,
            self.descriptor,
            float(self.descriptor.max_sprite_palette_distance),
        )
        self.assertLess(body_strong, 0.03)
        self.assertFalse(
            self.detector._passes_color_structure_gate(
                self.frame, self.descriptor, comp
            )
        )

    def test_wild_rose_fixture_two_rejects_poring(self) -> None:
        result = self.detector.detect(self.frame, "wild_rose")
        self.assertEqual(len(result.accepted), 2)
        for cand in result.accepted:
            self.assertGreater(
                abs(cand.center_x - 838) + abs(cand.center_y - 430),
                40,
                "Poring blob must not be accepted",
            )

    def test_foreign_wolf_crop_fails_wild_rose_palette(self) -> None:
        """Desert-wolf body colors are not Wild Rose palette — gate must reject."""
        wolf_path = (
            PROJECT_ROOT
            / "pybot"
            / "recognition"
            / "test-fixtures"
            / "game-screenshots"
            / "Wolf"
            / "1Wolf_Gray.png"
        )
        wolf_frame = cv2.imread(str(wolf_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(wolf_frame)
        wolf_frame = fixture_search_frame(wolf_frame)
        wolf_desc = self.detector.ensure_descriptor("desert_wolf")
        wolf_result = self.detector.detect(wolf_frame, "desert_wolf")
        self.assertGreaterEqual(len(wolf_result.accepted), 1)
        wolf = wolf_result.accepted[0]
        comp, crop = self._heat_crop(
            wolf_frame, wolf_desc, (wolf.center_x, wolf.center_y)
        )
        # Score the wolf heat-CC against Wild Rose descriptor.
        _p, _s, coverage, body_strong = required_groups_structure(
            crop,
            self.descriptor,
            float(self.descriptor.max_sprite_palette_distance),
        )
        self.assertTrue(
            coverage < 0.28 or body_strong < 0.03 or _s < 0.09 or _p < 2,
            f"wolf crop should look foreign to WR "
            f"(cov={coverage:.3f} body_strong={body_strong:.3f})",
        )
        self.assertFalse(
            self.detector._passes_color_structure_gate(
                wolf_frame, self.descriptor, comp
            )
        )


if __name__ == "__main__":
    unittest.main()
