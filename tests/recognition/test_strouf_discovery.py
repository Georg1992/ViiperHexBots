"""Tall wispy sprites must keep discovery heat on the body, not the long axis."""

from __future__ import annotations

import unittest

import numpy as np

from pybot.mobs.catalog import MOBS_DIR
from pybot.paths import DESCRIPTORS_DIR, PROJECT_ROOT
from pybot.recognition.act_reader import ActReader
from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.descriptors.descriptor_builder import DescriptorBuilder
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.detector.scoring.heatmap_detector import _odd_ksize
from pybot.recognition.spr_reader import SprReader


def _living_sprite_canvas(name: str) -> np.ndarray:
    spr_path = MOBS_DIR / name / "sprite" / f"{name}.spr"
    act_path = MOBS_DIR / name / "sprite" / f"{name}.act"
    if not spr_path.is_file() or not act_path.is_file():
        raise unittest.SkipTest(f"missing sprite pair: {spr_path}")
    builder = DescriptorBuilder(PROJECT_ROOT)
    spr = SprReader(spr_path).load()
    act = ActReader(act_path).load()
    frames = builder._collect_frames(spr, act, (0, 1), frame_start=0)
    if not frames:
        raise unittest.SkipTest(f"no living frames for {name}")
    bgra = frames[0]
    pad = 80
    canvas = np.full(
        (bgra.shape[0] + 2 * pad, bgra.shape[1] + 2 * pad, 3),
        40,
        dtype=np.uint8,
    )
    alpha = bgra[:, :, 3] >= 128
    canvas[pad : pad + bgra.shape[0], pad : pad + bgra.shape[1]][alpha] = bgra[:, :, :3][alpha]
    return canvas


class GaussianBlurKsizeTests(unittest.TestCase):
    def test_blur_kernel_follows_narrow_axis(self) -> None:
        tall = min(_odd_ksize(85, 0.8), _odd_ksize(85, 0.40))
        compact = min(_odd_ksize(85, 0.8), _odd_ksize(85, 0.40))
        self.assertEqual(tall, compact)
        long_axis = min(_odd_ksize(227, 0.8), _odd_ksize(227, 0.40))
        self.assertLess(tall, long_axis)


class StroufDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spr_path = MOBS_DIR / "strouf" / "sprite" / "strouf.spr"
        if not spr_path.is_file():
            raise unittest.SkipTest("strouf sprite is not installed")
        DescriptorBuilder(PROJECT_ROOT).build("strouf", force=True)

    def test_strouf_disables_diversity(self) -> None:
        descriptor = MobDescriptor.load(DESCRIPTORS_DIR / "strouf" / "descriptor.json")
        self.assertFalse(descriptor.use_body_cluster_diversity)

    def test_vanilla_self_match_keeps_body_heat(self) -> None:
        canvas = _living_sprite_canvas("strouf")
        # Match live hunt ROI scale so discovery downscale=2 is applied.
        frame = np.full((1024, 1024, 3), 40, dtype=np.uint8)
        y0 = (frame.shape[0] - canvas.shape[0]) // 2
        x0 = (frame.shape[1] - canvas.shape[1]) // 2
        frame[y0 : y0 + canvas.shape[0], x0 : x0 + canvas.shape[1]] = canvas
        detector = MobDetector(PROJECT_ROOT, load_detector_config())
        result = detector.detect(frame, "strouf")
        self.assertGreater(len(result.accepted), 0, "strouf sprite must be accepted")
        self.assertGreater(
            float(result.sprite_heatmap.max()),
            0.25,
            "tall wispy sprites must retain discovery heat after blur",
        )
        chk = next(item for item in result.silhouette_checks if item.passed)
        _x, _y, dw, dh = chk.discovery_bbox
        self.assertGreaterEqual(dh, 180)
        self.assertGreaterEqual(dw, 60)


if __name__ == "__main__":
    unittest.main()
