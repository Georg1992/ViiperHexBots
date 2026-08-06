"""Regression tests for static modified SPR/ACT generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pybot.recognition.act_reader import ActReader
from pybot.recognition.frame_renderer import render_act_frame
from pybot.recognition.spr_reader import SprReader
from scripts.make_mobs_big_red import (
    SCALE_FACTOR,
    _canonical_frame,
    _static_source_frame,
    process_mob_folder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = PROJECT_ROOT / "assets" / "mobs" / "Horn" / "sprite"


@unittest.skipUnless(
    (ASSET_DIR / "horn.spr").is_file() and (ASSET_DIR / "horn.act").is_file(),
    "Horn SPR/ACT assets not available",
)
class StaticModifiedSpriteTests(unittest.TestCase):
    def test_modified_pair_has_one_static_frame_and_preserves_action_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "modified"
            self.assertEqual(process_mob_folder(ASSET_DIR, output), 1)

            spr = SprReader(output / "horn.spr").load()
            act = ActReader(output / "horn.act").load()
            source = ActReader(ASSET_DIR / "horn.act").load()
            source_spr = SprReader(ASSET_DIR / "horn.spr").load()
            expected = render_act_frame(
                source_spr,
                _static_source_frame(_canonical_frame(source)),
            )

            self.assertEqual(spr.frame_count, 1)
            self.assertEqual(len(spr.indexed_frames), 1)
            self.assertEqual(len(spr.rgba_frames), 0)
            self.assertEqual(len(act.actions), len(source.actions))
            self.assertEqual(
                [len(action.frames) for action in act.actions],
                [len(action.frames) for action in source.actions],
            )

            living_end = max(1, len(act.actions) - 8)
            living_frames = [
                frame
                for action in act.actions[:living_end]
                for frame in action.frames
            ]
            self.assertTrue(living_frames)
            self.assertTrue(all(len(frame.layers) == 1 for frame in living_frames))
            self.assertEqual(
                {frame.layers[0].spr_frame_index for frame in living_frames},
                {0},
            )
            self.assertEqual(
                {frame.layers[0].color_tint for frame in living_frames},
                {(255, 255, 255, 255)},
            )
            self.assertEqual(
                {frame.layers[0].scale_x for frame in living_frames},
                {1.0},
            )

            death_frames = [
                frame
                for action in act.actions[living_end:]
                for frame in action.frames
            ]
            self.assertTrue(death_frames)
            self.assertEqual(
                {frame.layers[0].color_tint[3] for frame in death_frames},
                {0},
            )

            # The red/enlarged appearance is baked into the canonical SPR frame.
            frame = spr.get_frame(0)
            assert frame is not None
            opaque = frame.rgba[:, :, 3] >= 128
            self.assertTrue(opaque.any())
            self.assertTrue((frame.rgba[:, :, 2][opaque] > 0).any())
            self.assertEqual(int(frame.rgba[:, :, 0][opaque].max()), 0)
            self.assertEqual(int(frame.rgba[:, :, 1][opaque].max()), 0)
            self.assertEqual(frame.width, expected.shape[1])
            self.assertEqual(frame.height, expected.shape[0])
            source_frame = source_spr.get_frame(0)
            assert source_frame is not None
            self.assertGreaterEqual(frame.width, int(source_frame.width * SCALE_FACTOR))
            # Orientation is preserved: the generated frame keeps the same
            # top/bottom opacity ordering as the source frame (upright indexed
            # frames render correctly in the client; flipped RGBA did not).
            def _top_minus_bottom(image, h) -> float:
                return float(
                    image[: max(1, h // 4), :, 3].mean()
                    - image[-max(1, h // 4) :, :, 3].mean()
                )

            generated_orient = _top_minus_bottom(frame.rgba, frame.height)
            source_orient = _top_minus_bottom(source_frame.rgba, source_frame.height)
            self.assertEqual(
                generated_orient > 0.0,
                source_orient > 0.0,
                "generated frame orientation must match the source frame",
            )


if __name__ == "__main__":
    unittest.main()
