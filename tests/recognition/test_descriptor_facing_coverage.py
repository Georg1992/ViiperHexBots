"""Descriptor build must cover every stand/walk facing direction."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.act_reader import ActReader
from pybot.recognition.detector.descriptors.descriptor_builder import DescriptorBuilder
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.frame_renderer import render_act_frame
from pybot.recognition.spr_reader import SprReader


class DescriptorFacingCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = DescriptorBuilder(PROJECT_ROOT)
        cls.detector = MobDetector(PROJECT_ROOT, load_detector_config())
        cls.spr = SprReader(PROJECT_ROOT / "assets/mobs/Noxious/noxious.spr").load()
        cls.act = ActReader(PROJECT_ROOT / "assets/mobs/Noxious/noxious.act").load()
        cls.descriptor = cls.builder.build("noxious", force=True)

    def _canvas_for_action(self, action_index: int) -> tuple[np.ndarray, int, int]:
        ref = self.act.actions[action_index].frames[0]
        bgra = render_act_frame(self.spr, ref)
        ys, xs = np.where(bgra[:, :, 3] > 0)
        crop = bgra[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
        height, width = crop.shape[:2]
        pad = 30
        canvas = np.zeros((height + 2 * pad, width + 2 * pad, 3), dtype=np.uint8)
        canvas[:, :] = (40, 80, 40)
        y0, x0 = pad, pad
        for y in range(height):
            for x in range(width):
                if crop[y, x, 3] > 0:
                    canvas[y0 + y, x0 + x] = crop[y, x, :3]
        return canvas, x0 + width // 2, y0 + height // 2

    def test_descriptor_records_all_facing_structural_pixels(self) -> None:
        pairs = self.builder._living_facing_pairs(len(self.act.actions))
        self.assertEqual(len(pairs), 4)
        self.assertGreaterEqual(len(self.descriptor.structural_pixel_pairs()), 2)
        self.assertGreaterEqual(len(self.descriptor.match_palette_bgr), 11)

    def test_every_facing_passes_structural_gate(self) -> None:
        for action_index in range(8):
            canvas, cx, cy = self._canvas_for_action(action_index)
            hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
            score, bbox = self.detector._score_living_only_at(
                canvas,
                hsv,
                self.descriptor,
                cx,
                cy,
                0.65,
            )
            self.assertIsNotNone(bbox, msg=f"action {action_index} produced no bbox")
            assert bbox is not None
            self.assertTrue(
                self.detector._passes_structural_pixel_gate(canvas, self.descriptor, bbox),
                msg=f"action {action_index} failed structural gate",
            )
            self.assertIsNotNone(score)
            assert score is not None
            self.assertTrue(
                score.accepted,
                msg=f"action {action_index} rejected: {score.rejection_reason}",
            )


if __name__ == "__main__":
    unittest.main()
