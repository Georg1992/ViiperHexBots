"""Descriptor living-action selection: stand/walk + both jump rows (8-15)."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.act_reader import ActReader
from pybot.recognition.detector.descriptors.descriptor_builder import (
    JUMP_ACTION_COUNT,
    LIVING_ACTION_SIZE_TOLERANCE,
    LIVING_FACING_ACTION_LIMIT,
    STAND_WALK_ACTION_COUNT,
    DescriptorBuilder,
)
from pybot.recognition.detector.descriptors.layout_utils import (
    candidate_silhouette,
    silhouette_similarity,
)
from pybot.recognition.fixtures import shipped_mob_spr_stems
from pybot.recognition.frame_renderer import render_act_frame
from pybot.recognition.spr_reader import SprReader

SECOND_JUMP_PAIRS = ((12, 13), (14, 15))
ALL_LIVING_PAIRS = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 13),
    (14, 15),
)
SECOND_JUMP_ACTIONS = (12, 13, 14, 15)


class LivingFacingPairTests(unittest.TestCase):
    def test_limits_cover_stand_walk_and_both_jump_rows(self) -> None:
        self.assertEqual(STAND_WALK_ACTION_COUNT, 8)
        self.assertEqual(JUMP_ACTION_COUNT, 8)
        self.assertEqual(LIVING_FACING_ACTION_LIMIT, 16)

    def test_facing_pairs_for_full_ro_action_sheet(self) -> None:
        pairs = DescriptorBuilder._living_facing_pairs(40)
        self.assertEqual(pairs, ALL_LIVING_PAIRS)

    def test_facing_pairs_stop_at_available_actions(self) -> None:
        pairs_14 = DescriptorBuilder._living_facing_pairs(14)
        self.assertEqual(pairs_14[-1], (12, 13))
        self.assertNotIn((14, 15), pairs_14)

        pairs_12 = DescriptorBuilder._living_facing_pairs(12)
        self.assertEqual(pairs_12[-1], (10, 11))
        self.assertNotIn((12, 13), pairs_12)


class LivingActionPairMobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = DescriptorBuilder(PROJECT_ROOT)
        cls.mob_names = list(shipped_mob_spr_stems())

    @staticmethod
    def _pair_bbox_area(builder: DescriptorBuilder, spr, act, pair: tuple[int, int]) -> float | None:
        frames = builder._collect_frames(spr, act, pair, frame_start=0)
        if not frames:
            return None
        return float(frames[0].shape[0] * frames[0].shape[1])

    @classmethod
    def _mob_assets(cls, mob_name: str):
        asset_dir = cls.builder.asset_dir(mob_name)
        spr = SprReader(asset_dir / f"{mob_name}.spr").load()
        act = ActReader(asset_dir / f"{mob_name}.act").load()
        return spr, act

    @classmethod
    def _mobs_with_second_jump_row(cls) -> list[str]:
        included: list[str] = []
        for mob_name in cls.mob_names:
            spr, act = cls._mob_assets(mob_name)
            if len(act.actions) < LIVING_FACING_ACTION_LIMIT:
                continue
            pairs = cls.builder._living_action_pairs(act, spr)
            if all(jump_pair in pairs for jump_pair in SECOND_JUMP_PAIRS):
                included.append(mob_name)
        return included

    def test_second_jump_row_included_when_size_matches_baseline(self) -> None:
        self.assertGreater(len(self.mob_names), 0)
        for mob_name in self.mob_names:
            with self.subTest(mob=mob_name):
                spr, act = self._mob_assets(mob_name)
                if len(act.actions) < LIVING_FACING_ACTION_LIMIT:
                    self.skipTest(f"{mob_name} has fewer than {LIVING_FACING_ACTION_LIMIT} actions")

                raw_pairs = self.builder._living_facing_pairs(len(act.actions))
                pairs = self.builder._living_action_pairs(act, spr)
                self.assertEqual(pairs, raw_pairs[: len(pairs)])

                baseline = self._pair_bbox_area(self.builder, spr, act, (0, 1))
                if baseline is None:
                    self.skipTest(f"{mob_name} missing baseline stand/walk frames")
                min_area = baseline * (1.0 - LIVING_ACTION_SIZE_TOLERANCE)
                max_area = baseline * (1.0 + LIVING_ACTION_SIZE_TOLERANCE)

                for jump_pair in SECOND_JUMP_PAIRS:
                    if jump_pair not in raw_pairs:
                        continue
                    area = self._pair_bbox_area(self.builder, spr, act, jump_pair)
                    if area is None:
                        continue
                    if min_area <= area <= max_area:
                        self.assertIn(
                            jump_pair,
                            pairs,
                            f"{mob_name}: jump pair {jump_pair} matches stand/walk size",
                        )
                    else:
                        self.assertNotIn(
                            jump_pair,
                            pairs,
                            f"{mob_name}: jump pair {jump_pair} exceeds size tolerance",
                        )

    @staticmethod
    def _best_silhouette_similarity(
        candidate,
        facing_masks: list,
    ) -> float:
        best = 0.0
        for mask in facing_masks:
            if not mask.stable_mask or not any(mask.stable_mask):
                continue
            ref_avg = np.array(mask.avg_mask, dtype=np.float32).reshape(16, 16)
            ref_stable = np.array(mask.stable_mask, dtype=bool).reshape(16, 16)
            sim = silhouette_similarity(candidate, ref_avg, ref_stable)
            best = max(best, sim)
        return best

    def test_second_jump_row_contributes_silhouette_frames(self) -> None:
        mobs = self._mobs_with_second_jump_row()
        self.assertGreater(len(mobs), 0)
        for mob_name in mobs:
            with self.subTest(mob=mob_name):
                spr, act = self._mob_assets(mob_name)
                pairs = self.builder._living_action_pairs(act, spr)
                all_masks = self.builder._build_facing_silhouette_masks(spr, act, pairs)
                without_second_jump = tuple(p for p in pairs if p not in SECOND_JUMP_PAIRS)
                partial_masks = self.builder._build_facing_silhouette_masks(
                    spr, act, without_second_jump,
                )
                self.assertGreater(
                    len(all_masks),
                    len(partial_masks),
                    f"{mob_name}: second jump row must add facing silhouette refs",
                )


class SecondJumpSilhouetteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = DescriptorBuilder(PROJECT_ROOT)

    @staticmethod
    def _best_silhouette_similarity_for_action(
        facing_masks: list,
        descriptor,
        spr,
        act,
        action_index: int,
        frame_index: int = 0,
    ) -> float:
        palette = np.asarray(descriptor.match_palette_bgr, dtype=np.float32)
        target_w = max(8, int(round(descriptor.avg_width)))
        target_h = max(8, int(round(descriptor.avg_height)))

        bgra = render_act_frame(spr, act.actions[action_index].frames[frame_index])
        ys, xs = np.where(bgra[:, :, 3] > 0)
        crop = bgra[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1, :3]
        region = cv2.resize(crop, (target_w, target_h))
        candidate = candidate_silhouette(region, palette, 20.0, 16, 16)
        return LivingActionPairMobTests._best_silhouette_similarity(candidate, facing_masks)

    def test_second_jump_row_improves_silhouette_match(self) -> None:
        """Including jump row 2 in the refs must not reduce match for those poses."""
        mobs = LivingActionPairMobTests._mobs_with_second_jump_row()
        self.assertGreater(len(mobs), 0)
        for mob_name in mobs:
            with self.subTest(mob=mob_name):
                spr, act = LivingActionPairMobTests._mob_assets(mob_name)
                pairs = self.builder._living_action_pairs(act, spr)
                without_second_jump = tuple(p for p in pairs if p not in SECOND_JUMP_PAIRS)

                full_masks = self.builder._build_facing_silhouette_masks(spr, act, pairs)
                partial_masks = self.builder._build_facing_silhouette_masks(
                    spr, act, without_second_jump,
                )

                descriptor = self.builder.build(mob_name, force=False)
                for action_index in SECOND_JUMP_ACTIONS:
                    if action_index >= len(act.actions):
                        continue
                    full_sim = self._best_silhouette_similarity_for_action(
                        full_masks,
                        descriptor,
                        spr,
                        act,
                        action_index,
                    )
                    partial_sim = self._best_silhouette_similarity_for_action(
                        partial_masks,
                        descriptor,
                        spr,
                        act,
                        action_index,
                    )
                    if abs(full_sim - partial_sim) < 1e-6:
                        continue
                    if partial_sim < 0.50:
                        self.assertGreater(
                            full_sim,
                            partial_sim,
                            f"{mob_name} action {action_index}: "
                            f"full={full_sim:.3f} partial={partial_sim:.3f}",
                        )
                    else:
                        self.assertGreaterEqual(
                            full_sim,
                            partial_sim - 0.02,
                            f"{mob_name} action {action_index}: "
                            f"full={full_sim:.3f} partial={partial_sim:.3f}",
                        )


if __name__ == "__main__":
    unittest.main()
