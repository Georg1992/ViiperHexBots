"""Descriptor build and silhouette-gate field tests."""

from __future__ import annotations

import unittest

from pybot.paths import PROJECT_ROOT
from pybot.recognition.act_reader import ActReader
from pybot.recognition.detector.descriptors.descriptor_builder import DESCRIPTOR_VERSION, DescriptorBuilder
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
        self.assertGreater(len(descriptor.silhouette_masks), 0)
        self.assertLessEqual(len(descriptor.silhouette_masks), 4)
        self.assertEqual(len(descriptor.silhouette_masks[0].avg_mask), 256)

    def test_gate_masks_are_facing_medoids(self) -> None:
        asset_dir = self.builder.asset_dir("horn")
        spr = SprReader(asset_dir / "horn.spr").load()
        act = ActReader(asset_dir / "horn.act").load()
        facing_pairs = self.builder._living_action_pairs(act, spr)
        facing_masks = self.builder._build_facing_silhouette_masks(spr, act, facing_pairs)
        descriptor = self.builder.build("horn", force=True)
        self.assertEqual(len(descriptor.silhouette_masks), 4)
        for gate_mask in descriptor.silhouette_masks:
            self.assertIn(gate_mask, facing_masks)
        all_union = self.builder._merge_facing_silhouette_masks(facing_masks)
        max_gate_stable = max(sum(mask.stable_mask) for mask in descriptor.silhouette_masks)
        self.assertLess(max_gate_stable, sum(all_union.stable_mask))


if __name__ == "__main__":
    unittest.main()
