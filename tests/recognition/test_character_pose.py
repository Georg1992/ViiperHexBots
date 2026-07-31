"""Center character pose: sitting is shorter than standing; falcon is ignored."""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2

from pybot.recognition.ui.character_pose import (
    CharacterPose,
    check_is_sitting,
    check_is_standing,
    measure_center_pose,
)

_TESTS = Path(__file__).resolve().parents[1]
_SIT = [_TESTS / "sit.png", _TESTS / "sit2.png", _TESTS / "sit3.png", _TESTS / "sit4.png"]
_STAND = [_TESTS / "stand.png", _TESTS / "Stand2.png", _TESTS / "stand3.png", _TESTS / "stand4.png"]


class CharacterPoseTests(unittest.TestCase):
    def test_sit_shorter_than_stand(self) -> None:
        sit_heights = []
        stand_heights = []
        for path in _SIT:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            self.assertIsNotNone(img, msg=str(path))
            pose = measure_center_pose(img)
            self.assertIsNotNone(pose, msg=str(path))
            assert pose is not None
            sit_heights.append(pose.body_height)
            self.assertTrue(check_is_sitting(pose), msg=str(path))
            self.assertFalse(check_is_standing(pose), msg=str(path))
        for path in _STAND:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            self.assertIsNotNone(img, msg=str(path))
            pose = measure_center_pose(img)
            self.assertIsNotNone(pose, msg=str(path))
            assert pose is not None
            stand_heights.append(pose.body_height)
            self.assertTrue(check_is_standing(pose), msg=str(path))
            self.assertFalse(check_is_sitting(pose), msg=str(path))

        self.assertLess(max(sit_heights), min(stand_heights))
        # Falcon must not erase the sit/stand gap (body run, not full stack).
        self.assertGreaterEqual(min(stand_heights) - max(sit_heights), 10)

    def test_falcon_merge_band_is_ambiguous_not_standing(self) -> None:
        """Heights where falcon glues onto a sit must not read as standing."""
        for height in (77, 80, 85):
            pose = CharacterPose(body_height=height, fg_count=1000)
            self.assertIsNone(check_is_sitting(pose), msg=height)
            self.assertIsNone(check_is_standing(pose), msg=height)
        self.assertTrue(check_is_sitting(CharacterPose(76, 1)))
        self.assertTrue(check_is_standing(CharacterPose(86, 1)))


if __name__ == "__main__":
    unittest.main()
