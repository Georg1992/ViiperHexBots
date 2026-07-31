"""Mob behavior registry and UI labels."""

from __future__ import annotations

import unittest

from pybot.runtime.mob_behaviors import AnubisBehavior, get_mob_behavior


class MobBehaviorUiLabelTests(unittest.TestCase):
    def test_anubis_marked_special_kiting(self) -> None:
        behavior = get_mob_behavior("anubis")
        self.assertIsInstance(behavior, AnubisBehavior)
        self.assertTrue(behavior.has_custom_behavior())
        self.assertEqual(behavior.special_behavior_label(), "kiting")

    def test_default_mob_has_no_special_label(self) -> None:
        behavior = get_mob_behavior("horn")
        self.assertFalse(behavior.has_custom_behavior())
        self.assertIsNone(behavior.special_behavior_label())


if __name__ == "__main__":
    unittest.main()
