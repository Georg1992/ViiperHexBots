"""Descriptor build and silhouette-gate field tests."""

from __future__ import annotations

import unittest

import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.act_reader import ActReader
from pybot.recognition.detector.descriptors.descriptor_builder import (
    ACTIONS_PER_ANIMATION,
    DEAD_ACTION_COUNT,
    DESCRIPTOR_VERSION,
    DescriptorBuilder,
    GATE_SILHOUETTE_REF_COUNTS,
    MAX_DEATH_GATE_SILHOUETTE_MASKS,
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
            len(descriptor.death_silhouette_masks), 1
        )
        self.assertLessEqual(
            len(descriptor.death_silhouette_masks), MAX_DEATH_GATE_SILHOUETTE_MASKS
        )
        self.assertEqual(len(descriptor.death_silhouette_masks[0].avg_mask), 256)
        # Death refs are farthest-from-living: every kept mask must differ
        # from the living gate set (no identical living pose as a death ref).
        living_avgs = {tuple(m.avg_mask) for m in descriptor.silhouette_masks}
        death_avgs = {tuple(m.avg_mask) for m in descriptor.death_silhouette_masks}
        distinct = death_avgs - living_avgs
        self.assertEqual(
            len(distinct),
            len(death_avgs),
            "death gate refs must not reuse living silhouette masks",
        )

    def test_death_action_indices_are_act_editor_die_group(self) -> None:
        # 5 animations × 8 dirs → Die = animation 4 → actions 32..39
        self.assertEqual(
            DescriptorBuilder._death_action_indices(40),
            (32, 33, 34, 35, 36, 37, 38, 39),
        )
        # 6 animations × 8 dirs → Die = animation 5 → actions 40..47
        self.assertEqual(
            DescriptorBuilder._death_action_indices(48),
            (40, 41, 42, 43, 44, 45, 46, 47),
        )
        self.assertEqual(DescriptorBuilder._death_action_indices(16), ())
        self.assertEqual(DescriptorBuilder._death_action_indices(41), ())
        self.assertEqual(ACTIONS_PER_ANIMATION, 8)
        self.assertEqual(DEAD_ACTION_COUNT, 8)

    def test_death_gate_picks_farthest_from_living_capped(self) -> None:
        asset_dir = self.builder.asset_dir("horn")
        spr = SprReader(asset_dir / "horn.spr").load()
        act = ActReader(asset_dir / "horn.act").load()
        facing_pairs = self.builder._living_action_pairs(act, spr)
        living = self.builder._build_frame_silhouette_masks(spr, act, facing_pairs)
        death_masks = self.builder._build_death_gate_silhouette_masks(
            spr, act, living_masks=living,
        )
        self.assertGreater(len(death_masks), 0)
        self.assertLessEqual(len(death_masks), MAX_DEATH_GATE_SILHOUETTE_MASKS)

        descriptor = self.builder.build("horn", force=True)
        death = descriptor.death_silhouette_masks
        self.assertEqual(len(death), len(death_masks))
        death_avgs = {tuple(m.avg_mask) for m in death_masks}
        for mask in death:
            self.assertIn(tuple(mask.avg_mask), death_avgs)

        # First pick is the global farthest from living among the Die pool.
        if living and death_masks:
            living_avgs = [
                np.asarray(m.avg_mask, dtype=np.float32) for m in living
            ]
            first = np.asarray(death_masks[0].avg_mask, dtype=np.float32)
            first_sim = max(
                self.builder._soft_jaccard(first, la) for la in living_avgs
            )
            pool = self.builder._build_death_frame_silhouette_masks(
                spr, act, self.builder._death_action_indices(len(act.actions)),
            )
            pool_sims = []
            for mask in pool:
                avg = np.asarray(mask.avg_mask, dtype=np.float32)
                pool_sims.append(
                    max(self.builder._soft_jaccard(avg, la) for la in living_avgs)
                )
            self.assertAlmostEqual(first_sim, min(pool_sims), places=5)

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
