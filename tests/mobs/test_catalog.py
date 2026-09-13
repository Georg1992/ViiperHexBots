"""Catalog grouping for the Isilla + Vanberk hunt option."""

from __future__ import annotations

import unittest

from pybot.mobs.catalog import (
    ISILLA_VANBERK_DISPLAY,
    ISILLA_VANBERK_MEMBERS,
    ISILLA_VANBERK_OPTION,
    MobEntry,
    collapse_isilla_vanberk,
    hunt_mob_names,
    is_builtin_mob,
)
from pybot.paths import MOBS_DIR


def _entry(name: str) -> MobEntry:
    return MobEntry(asset_name=name, display_name=name, descriptor_name=name)


class HuntMobNamesTests(unittest.TestCase):
    def test_single_mob_is_itself(self) -> None:
        self.assertEqual(hunt_mob_names("horn"), ("horn",))

    def test_pair_option_returns_both_sprites(self) -> None:
        self.assertEqual(hunt_mob_names(ISILLA_VANBERK_OPTION), ISILLA_VANBERK_MEMBERS)

    def test_pair_members_and_option_are_builtin(self) -> None:
        self.assertTrue(is_builtin_mob("isilla"))
        self.assertTrue(is_builtin_mob("vanberk"))
        self.assertTrue(is_builtin_mob(ISILLA_VANBERK_OPTION))

    def test_empty_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            hunt_mob_names("  ")

    def test_shipped_sprite_pairs_exist(self) -> None:
        for name in ISILLA_VANBERK_MEMBERS:
            sprite = MOBS_DIR / name / "sprite" / f"{name}.spr"
            act = MOBS_DIR / name / "sprite" / f"{name}.act"
            self.assertTrue(sprite.is_file(), sprite)
            self.assertTrue(act.is_file(), act)


class CollapseIsillaVanberkTests(unittest.TestCase):
    def test_both_present_become_one_option(self) -> None:
        catalog = collapse_isilla_vanberk(
            [_entry("horn"), _entry("isilla"), _entry("vanberk")]
        )
        names = [entry.descriptor_name for entry in catalog]
        self.assertEqual(names, ["horn", ISILLA_VANBERK_OPTION])
        self.assertEqual(catalog[1].display_name, ISILLA_VANBERK_DISPLAY)
        self.assertTrue(catalog[1].is_builtin)

    def test_only_isilla_stays_separate(self) -> None:
        catalog = collapse_isilla_vanberk([_entry("isilla"), _entry("horn")])
        self.assertEqual(
            [entry.descriptor_name for entry in catalog],
            ["isilla", "horn"],
        )

    def test_only_vanberk_stays_separate(self) -> None:
        catalog = collapse_isilla_vanberk([_entry("vanberk")])
        self.assertEqual([entry.descriptor_name for entry in catalog], ["vanberk"])


if __name__ == "__main__":
    unittest.main()
