"""Mob behavior registry — custom behavior is Anubis-only."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pybot.mobs.catalog import load_mob_catalog
from pybot.runtime.mob_behaviors import (
    AnubisBehavior,
    MobBehavior,
    _BEHAVIOR_REGISTRY,
    get_mob_behavior,
    mob_has_custom_behavior,
)


class MobBehaviorRegistryTests(unittest.TestCase):
    def test_registry_contains_only_anubis(self) -> None:
        self.assertEqual(set(_BEHAVIOR_REGISTRY.keys()), {"anubis"})
        self.assertIsInstance(_BEHAVIOR_REGISTRY["anubis"], AnubisBehavior)

    def test_anubis_has_custom_behavior(self) -> None:
        self.assertTrue(mob_has_custom_behavior("anubis"))
        self.assertTrue(mob_has_custom_behavior("Anubis"))
        self.assertIsInstance(get_mob_behavior("anubis"), AnubisBehavior)

    def test_other_catalog_mobs_have_no_custom_behavior(self) -> None:
        catalog = load_mob_catalog(ensure_assets=False)
        self.assertGreater(len(catalog), 1)
        for mob in catalog:
            if mob.descriptor_name.lower() == "anubis":
                self.assertTrue(mob_has_custom_behavior(mob.descriptor_name))
                continue
            self.assertFalse(
                mob_has_custom_behavior(mob.descriptor_name),
                msg=f"{mob.descriptor_name} must not have custom behavior",
            )
            self.assertIsInstance(get_mob_behavior(mob.descriptor_name), MobBehavior)
            self.assertNotIsInstance(
                get_mob_behavior(mob.descriptor_name), AnubisBehavior
            )

    def test_unknown_mob_has_no_custom_behavior(self) -> None:
        self.assertFalse(mob_has_custom_behavior("horn"))
        self.assertFalse(mob_has_custom_behavior(""))
        self.assertIsInstance(get_mob_behavior("not_a_mob"), MobBehavior)

    def test_anubis_kite_moves_away_from_mob_center(self) -> None:
        behavior = AnubisBehavior()
        backend = MagicMock()
        backend.left_click.return_value = True
        # Char at (100, 100); one mob at (120, 100) → kite left to (80, 100).
        ok = behavior.kite_after_attack(
            100, 100, backend, all_mobs=[(120, 100)],
        )
        self.assertTrue(ok)
        backend.move_mouse.assert_called_once_with(80, 100)
        backend.left_click.assert_called_once()


if __name__ == "__main__":
    unittest.main()
