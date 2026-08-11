"""Descriptor build and silhouette-gate field tests."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.act_reader import ActReader
from pybot.recognition.detector.descriptors.descriptor import SizeDescriptor
from pybot.recognition.detector.descriptors.descriptor_builder import (
    DESCRIPTOR_VERSION,
    DescriptorBuilder,
    GATE_SILHOUETTE_REF_COUNTS,
    MIN_GATE_SILHOUETTE_MASKS,
)
from pybot.recognition.detector.detector import MobDetector, load_detector_config
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

    def test_aspect_band_uses_descriptor_normalized_units(self) -> None:
        """Build-time aspect bounds match the runtime geometry coordinate system."""
        frame = np.zeros((20, 10, 4), dtype=np.uint8)
        frame[:, :, 3] = 255
        band = self.builder._measure_aspect_band(
            [frame],
            margin=0.15,
            reference_aspect=0.5,
        )
        self.assertEqual(band, (0.85, 1.15))

        descriptor = self.builder.build("horn", force=True)
        reference_aspect = descriptor.avg_width / descriptor.avg_height
        self.assertLessEqual(descriptor.min_aspect_ratio, 1.0)
        self.assertGreaterEqual(descriptor.max_aspect_ratio, 1.0)
        self.assertGreater(reference_aspect, 0.0)

    def test_runtime_geometry_consumes_normalized_aspect_band(self) -> None:
        """The runtime gate accepts a normalized in-band shape, not raw bounds."""
        base = self.builder.build("horn", force=True)
        descriptor = replace(
            base,
            size=SizeDescriptor(avg_width=10.0, avg_height=20.0),
            min_aspect_ratio=0.85,
            max_aspect_ratio=1.15,
        )
        detector = MobDetector(PROJECT_ROOT, load_detector_config())

        # 11x20 has raw aspect .55, but normalized aspect 1.10: in band.
        self.assertTrue(
            detector._passes_size_aspect_vs_descriptor(
                11, 20, descriptor, require_min_area=False,
            )
        )
        # 10x10 has normalized aspect 2.0: outside the same band.
        self.assertFalse(
            detector._passes_size_aspect_vs_descriptor(
                10, 10, descriptor, require_min_area=False,
            )
        )

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
