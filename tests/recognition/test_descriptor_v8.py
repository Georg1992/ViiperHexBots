"""Descriptor v8 build and backward compatibility tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from pybot.paths import PROJECT_ROOT
from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.descriptors.descriptor_builder import DESCRIPTOR_VERSION, DescriptorBuilder


class DescriptorV8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = DescriptorBuilder(PROJECT_ROOT)

    def test_builds_v8_fields(self) -> None:
        descriptor = self.builder.build("horn", force=True)
        self.assertEqual(descriptor.version, DESCRIPTOR_VERSION)
        self.assertIsNotNone(descriptor.size_stats)
        self.assertIsNotNone(descriptor.occupancy_stats)
        self.assertGreater(len(descriptor.color_stats), 0)
        self.assertIsNotNone(descriptor.layout_grid)
        self.assertIsNotNone(descriptor.silhouette_mask)
        assert descriptor.layout_grid is not None
        self.assertEqual(descriptor.layout_grid.grid_size, 5)
        self.assertEqual(len(descriptor.layout_grid.avg_occupancy), 25)
        assert descriptor.silhouette_mask is not None
        self.assertEqual(len(descriptor.silhouette_mask.avg_mask), 256)

    def test_v7_json_still_loads(self) -> None:
        payload = {
            "mobName": "legacy",
            "version": 7,
            "size": {"avg_width": 40.0, "avg_height": 35.0},
            "dominantColor": {
                "label": "dominant",
                "bgr": [10.0, 20.0, 30.0],
                "hsv": [100.0, 120.0, 80.0],
                "fraction": 0.4,
                "tolerance": [12, 35, 55],
            },
            "supportingColors": [],
            "accentColors": [
                {
                    "label": "accent_0",
                    "bgr": [200.0, 180.0, 20.0],
                    "hsv": [20.0, 200.0, 200.0],
                    "fraction": 0.1,
                    "tolerance": [16, 60, 65],
                }
            ],
            "rareColors": [],
            "spritePaletteBgr": [[10, 20, 30]],
            "matchPaletteBgr": [[10, 20, 30]],
            "hsvHistogram": [0.0] * 384,
        }
        descriptor = MobDescriptor.from_dict(payload)
        self.assertEqual(descriptor.version, 7)
        self.assertIsNone(descriptor.layout_grid)
        stats = descriptor.effective_size_stats()
        self.assertGreater(stats.max_width, stats.min_width)


if __name__ == "__main__":
    unittest.main()
