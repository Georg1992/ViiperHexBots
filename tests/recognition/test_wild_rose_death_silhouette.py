"""Wild Rose fixtures: death silhouette must hit corpses, not living roses."""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2

from pybot.paths import PROJECT_ROOT
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.fixtures import fixture_search_frame


_FIXTURE = (
    PROJECT_ROOT
    / "pybot"
    / "recognition"
    / "test-fixtures"
    / "game-screenshots"
    / "WildRose"
    / "2WildRose_Gray2.png"
)


class WildRoseDeathSilhouetteFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.detector = MobDetector(PROJECT_ROOT, load_detector_config())
        cls.descriptor = cls.detector.ensure_descriptor("wild_rose")
        raw = cv2.imread(str(_FIXTURE))
        assert raw is not None, f"missing fixture {_FIXTURE}"
        cls.frame = fixture_search_frame(raw)

    def test_living_accepts_are_not_death_wins(self) -> None:
        result = self.detector.detect(self.frame, "wild_rose")
        self.assertEqual(len(result.accepted), 2)
        for cand in result.accepted:
            wins = self.detector.death_wins_living_at(
                self.frame,
                self.descriptor,
                cand.center_x,
                cand.center_y,
                cand.candidate_scale or 1.0,
            )
            self.assertFalse(
                wins,
                f"living rose at ({cand.center_x},{cand.center_y}) must not "
                "confirm death",
            )

    def test_corpse_peak_is_death_wins(self) -> None:
        heat = self.detector.heatmap_detector.build_sprite_heatmap(
            self.frame, self.descriptor,
        )
        blobs = self.detector.heatmap_detector.top_centers(heat, self.descriptor)
        corpse_hits = []
        for cx, cy, _heat, _bbox in blobs:
            if self.detector.death_wins_living_at(
                self.frame, self.descriptor, cx, cy, 1.0,
            ):
                corpse_hits.append((cx, cy))
        self.assertGreaterEqual(
            len(corpse_hits),
            1,
            "2WildRose_Gray2 must have at least one death-winning corpse peak",
        )


if __name__ == "__main__":
    unittest.main()
