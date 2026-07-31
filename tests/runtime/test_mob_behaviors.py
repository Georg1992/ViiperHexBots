"""Mob behavior registry and UI labels."""

from __future__ import annotations

import unittest

from pybot.runtime.mob_behaviors import AnubisBehavior, get_mob_behavior


class MobBehaviorUiLabelTests(unittest.TestCase):
    def test_anubis_marked_custom_behavior(self) -> None:
        behavior = get_mob_behavior("anubis")
        self.assertIsInstance(behavior, AnubisBehavior)
        self.assertTrue(behavior.has_custom_behavior())

    def test_default_mob_has_no_custom_behavior(self) -> None:
        behavior = get_mob_behavior("horn")
        self.assertFalse(behavior.has_custom_behavior())


if __name__ == "__main__":
    unittest.main()
