"""Regression tests for static modified SPR/ACT generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pybot.recognition.act_reader import (
    ActAction,
    ActFile,
    ActFrameRef,
    ActReader,
    ActSpriteLayer,
)
from pybot.recognition.frame_renderer import render_act_frame
from pybot.recognition.spr_reader import SprReader
from scripts.make_mobs_big_red import (
    SCALE_FACTOR,
    _canonical_frame,
    _dead_actions,
    _static_source_frame,
    make_static_act_bytes,
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

            dead_actions = _dead_actions(len(act.actions))
            living_frames = [
                frame
                for action in act.actions
                if action.index not in dead_actions
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
                for action in act.actions
                if action.index in dead_actions
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


class DeadActionRangeTests(unittest.TestCase):
    def test_dead_actions_are_fixed_32_39_not_last_eight(self) -> None:
        """Death is always actions 32-39 regardless of total action count."""
        self.assertEqual(_dead_actions(48), set(range(32, 40)))
        self.assertEqual(_dead_actions(40), set(range(32, 40)))
        # A truncated layout exposes only the death actions it contains.
        self.assertEqual(_dead_actions(36), set(range(32, 36)))
        self.assertEqual(_dead_actions(31), set())
        self.assertEqual(_dead_actions(0), set())

    def _fake_act(self, num_actions: int) -> ActFile:
        layer = ActSpriteLayer(
            x=0,
            y=0,
            spr_frame_index=0,
            mirror=False,
            color_tint=(255, 255, 255, 255),
            scale_x=1.0,
            scale_y=1.0,
            rotation=0.0,
            image_type=0,
        )

        def frame(index: int) -> ActFrameRef:
            return ActFrameRef(
                spr_frame_index=0,
                delay_ms=0,
                action_index=index,
                frame_index=0,
                layers=[layer],
            )

        return ActFile(
            path=Path("<synthetic>"),
            version=0x0200,
            actions=[
                ActAction(name=f"action_{i}", index=i, frames=[frame(i)])
                for i in range(num_actions)
            ],
        )

    def test_static_act_transparent_only_actions_32_39(self) -> None:
        """Specials after 39 (e.g. 40-47) stay visible; only 32-39 are death."""
        out, stats = make_static_act_bytes(
            self._fake_act(48),
            origin=(0, 0),
            width=1,
            height=1,
        )
        self.assertEqual(stats["dead_actions"], list(range(32, 40)))
        with tempfile.TemporaryDirectory() as tmp:
            act_path = Path(tmp) / "synthetic.act"
            act_path.write_bytes(out)
            parsed = ActReader(act_path).load()
            self.assertEqual(len(parsed.actions), 48)
            for action in parsed.actions:
                alpha = action.frames[0].layers[0].color_tint[3]
                if 32 <= action.index < 40:
                    self.assertEqual(
                        alpha, 0, f"death action {action.index} must be transparent"
                    )
                else:
                    self.assertEqual(
                        alpha, 255, f"living action {action.index} must stay visible"
                    )


if __name__ == "__main__":
    unittest.main()
