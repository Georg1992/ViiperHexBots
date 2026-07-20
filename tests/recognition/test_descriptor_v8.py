"""Descriptor build and silhouette-gate field tests."""

from __future__ import annotations

import unittest

from pybot.paths import PROJECT_ROOT
from pybot.recognition.act_reader import ActReader
from pybot.recognition.detector.descriptors.descriptor_builder import (
    DESCRIPTOR_VERSION,
    DescriptorBuilder,
    GATE_SILHOUETTE_REF_COUNTS,
    MIN_DEATH_GATE_SILHOUETTE_MASKS,
    MIN_GATE_SILHOUETTE_MASKS,
)
from pybot.recognition.spr_reader import SprReader


class DescriptorV8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = DescriptorBuilder(PROJECT_ROOT)

    def test_builds_runtime_fields(self) -> None:
        descriptor = self.builder.build("horn", force=True)
        self.assertEqual(descriptor.version, DESCRIPTOR_VERSION)
        self.assertGreater(descriptor.avg_width, 0)
        self.assertGreater(descriptor.avg_height, 0)
        self.assertGreater(len(descriptor.match_palette_bgr), 0)
        self.assertEqual(len(descriptor.match_palette_weights), len(descriptor.match_palette_bgr))
        self.assertGreater(len(descriptor.accent_colors), 0)
        self.assertGreater(len(descriptor.dominant_pixels_bgr), 0)
        self.assertGreater(len(descriptor.accent_pixels_bgr), 0)
        self.assertGreaterEqual(len(descriptor.silhouette_masks), MIN_GATE_SILHOUETTE_MASKS)
        self.assertIn(len(descriptor.silhouette_masks), GATE_SILHOUETTE_REF_COUNTS)
        self.assertEqual(len(descriptor.silhouette_masks[0].avg_mask), 256)
        self.assertGreaterEqual(
            len(descriptor.death_silhouette_masks), MIN_DEATH_GATE_SILHOUETTE_MASKS
        )
        self.assertEqual(len(descriptor.death_silhouette_masks[0].avg_mask), 256)
        living_avgs = {tuple(m.avg_mask) for m in descriptor.silhouette_masks}
        death_avgs = {tuple(m.avg_mask) for m in descriptor.death_silhouette_masks}
        self.assertFalse(living_avgs == death_avgs)

    def test_death_action_indices_use_last_eight(self) -> None:
        self.assertEqual(
            DescriptorBuilder._death_action_indices(40),
            (32, 33, 34, 35, 36, 37, 38, 39),
        )
        self.assertEqual(DescriptorBuilder._death_action_indices(16), ())

    def test_gate_masks_are_selected_from_frames(self) -> None:
        asset_dir = self.builder.asset_dir("horn")
        spr = SprReader(asset_dir / "horn.spr").load()
        act = ActReader(asset_dir / "horn.act").load()
        facing_pairs = self.builder._living_action_pairs(act, spr)
        frame_masks = self.builder._build_frame_silhouette_masks(spr, act, facing_pairs)
        descriptor = self.builder.build("horn", force=True)
        self.assertGreater(len(frame_masks), 0)
        self.assertLessEqual(len(descriptor.silhouette_masks), len(frame_masks))
        frame_avgs = {tuple(mask.avg_mask) for mask in frame_masks}
        for gate_mask in descriptor.silhouette_masks:
            self.assertIn(tuple(gate_mask.avg_mask), frame_avgs)


if __name__ == "__main__":
    unittest.main()
