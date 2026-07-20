"""ACT default tint must not destroy sprite colors."""

from __future__ import annotations

import unittest

import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.act_reader import ActReader
from pybot.recognition.frame_renderer import render_act_frame
from pybot.recognition.spr_reader import SprReader


class FrameRendererTintTests(unittest.TestCase):
    def test_default_act_tint_preserves_sprite_channels(self) -> None:
        spr = SprReader(PROJECT_ROOT / "assets/mobs/TharaFrog/thara_frog.spr").load()
        act = ActReader(PROJECT_ROOT / "assets/mobs/TharaFrog/thara_frog.act").load()
        layer = act.actions[0].frames[0].layers[0]
        self.assertEqual(layer.color_tint, (255, 255, 255, 255))

        raw = spr.get_frame(layer.spr_frame_index).rgba
        rendered = render_act_frame(spr, act.actions[0].frames[0])

        raw_visible = raw[raw[:, :, 3] > 0][:, :3]
        rendered_visible = rendered[rendered[:, :, 3] > 0][:, :3]
        self.assertGreater(int(np.sum(raw_visible[:, 1] > 0)), 0)
        self.assertGreater(int(np.sum(rendered_visible[:, 1] > 0)), 0)
        self.assertGreater(int(np.sum(rendered_visible[:, 2] > 0)), 0)


if __name__ == "__main__":
    unittest.main()
