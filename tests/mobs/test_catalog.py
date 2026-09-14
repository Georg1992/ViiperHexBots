"""Catalog grouping for similar-sprite hunt options."""

from __future__ import annotations

import unittest

from pybot.mobs.catalog import (
    ISILLA_VANBERK_DISPLAY,
    ISILLA_VANBERK_MEMBERS,
    ISILLA_VANBERK_OPTION,
    MERMAN_STROUF_DISPLAY,
    MERMAN_STROUF_MEMBERS,
    MERMAN_STROUF_OPTION,
    MobEntry,
    collapse_isilla_vanberk,
    collapse_merman_strouf,
    collapse_paired_hunts,
    hunt_color_key,
    hunt_mob_names,
    is_builtin_mob,
)
from pybot.mobs.marker_sprites import modified_sprite_rgb
from pybot.paths import MOBS_DIR


def _entry(name: str) -> MobEntry:
    return MobEntry(asset_name=name, display_name=name, descriptor_name=name)


class HuntMobNamesTests(unittest.TestCase):
    def test_single_mob_is_itself(self) -> None:
        self.assertEqual(hunt_mob_names("horn"), ("horn",))

    def test_pair_option_returns_both_sprites(self) -> None:
        self.assertEqual(hunt_mob_names(ISILLA_VANBERK_OPTION), ISILLA_VANBERK_MEMBERS)
        self.assertEqual(hunt_mob_names(MERMAN_STROUF_OPTION), MERMAN_STROUF_MEMBERS)

    def test_pair_members_and_option_are_builtin(self) -> None:
        for name in (
            *ISILLA_VANBERK_MEMBERS,
            ISILLA_VANBERK_OPTION,
            *MERMAN_STROUF_MEMBERS,
            MERMAN_STROUF_OPTION,
        ):
            with self.subTest(name=name):
                self.assertTrue(is_builtin_mob(name))

    def test_empty_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            hunt_mob_names("  ")

    def test_shipped_sprite_pairs_exist(self) -> None:
        for name in (*ISILLA_VANBERK_MEMBERS, *MERMAN_STROUF_MEMBERS):
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


class CollapseMermanStroufTests(unittest.TestCase):
    def test_both_present_become_one_option(self) -> None:
        catalog = collapse_merman_strouf(
            [_entry("horn"), _entry("merman"), _entry("strouf")]
        )
        names = [entry.descriptor_name for entry in catalog]
        self.assertEqual(names, ["horn", MERMAN_STROUF_OPTION])
        self.assertEqual(catalog[1].display_name, MERMAN_STROUF_DISPLAY)
        self.assertTrue(catalog[1].is_builtin)

    def test_only_merman_stays_separate(self) -> None:
        catalog = collapse_merman_strouf([_entry("merman"), _entry("horn")])
        self.assertEqual(
            [entry.descriptor_name for entry in catalog],
            ["merman", "horn"],
        )

    def test_only_strouf_stays_separate(self) -> None:
        catalog = collapse_merman_strouf([_entry("strouf")])
        self.assertEqual([entry.descriptor_name for entry in catalog], ["strouf"])


class CollapsePairedHuntsTests(unittest.TestCase):
    def test_both_pairs_collapse_independently(self) -> None:
        catalog = collapse_paired_hunts(
            [
                _entry("horn"),
                _entry("isilla"),
                _entry("vanberk"),
                _entry("merman"),
                _entry("strouf"),
            ]
        )
        names = [entry.descriptor_name for entry in catalog]
        self.assertEqual(names, ["horn", ISILLA_VANBERK_OPTION, MERMAN_STROUF_OPTION])
        self.assertEqual(catalog[1].display_name, ISILLA_VANBERK_DISPLAY)
        self.assertEqual(catalog[2].display_name, MERMAN_STROUF_DISPLAY)


class HuntColorKeyTests(unittest.TestCase):
    def test_pair_members_share_a_color_key(self) -> None:
        self.assertEqual(hunt_color_key("isilla"), ISILLA_VANBERK_OPTION)
        self.assertEqual(hunt_color_key("vanberk"), ISILLA_VANBERK_OPTION)
        self.assertEqual(hunt_color_key("merman"), MERMAN_STROUF_OPTION)
        self.assertEqual(hunt_color_key("strouf"), MERMAN_STROUF_OPTION)
        self.assertEqual(hunt_color_key("horn"), "horn")

    def test_pair_members_share_marker_color_and_other_hunts_differ(self) -> None:
        isilla = modified_sprite_rgb("isilla")
        vanberk = modified_sprite_rgb("vanberk")
        merman = modified_sprite_rgb("merman")
        strouf = modified_sprite_rgb("strouf")
        horn = modified_sprite_rgb("horn")
        self.assertEqual(isilla, vanberk)
        self.assertEqual(merman, strouf)
        self.assertNotEqual(isilla, merman)
        self.assertNotEqual(isilla, horn)
        self.assertNotEqual(merman, horn)
        self.assertEqual(modified_sprite_rgb(ISILLA_VANBERK_OPTION), isilla)
        self.assertEqual(modified_sprite_rgb(MERMAN_STROUF_OPTION), merman)

    def test_empty_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            hunt_color_key("  ")


if __name__ == "__main__":
    unittest.main()
