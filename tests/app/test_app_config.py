"""Application layer tests."""

from __future__ import annotations

import configparser
import tempfile
import unittest
from pathlib import Path

from pybot.app.config_store import AppConfig, list_client_profiles
from pybot.config.clients import memory_reading_enabled
from pybot.config.runtime import hunt_runtime_config_from_settings, resolve_mob_name
from pybot.config.schema import AppSettings, MobCustomSettings
from pybot.mobs.catalog import load_mob_catalog, mob_display_name
from pybot.paths import PROJECT_ROOT


class AppConfigTests(unittest.TestCase):
    def test_round_trip_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.ini"
            config = AppConfig(config_path=path)
            config.window_id = 123
            config.window_title = "Test Game"
            config.window_process = "client.exe"
            config.skill_button = "e"
            config.teleport_button = "q"
            config.search_range = 12
            config.hunt_mode = "walk"
            config.mob_custom_settings["horn"] = MobCustomSettings(
                kiting_tick_s=0.5,
                kite_distance_cells=7,
                debuff_button="r",
                heal_button="q",
                buff1_button="f1",
                buff1_delay_s=10,
            )
            config.save()

            loaded = AppConfig(config_path=path).load()
            self.assertEqual(loaded.window_id, 123)
            self.assertEqual(loaded.window_title, "Test Game")
            self.assertEqual(loaded.search_range, 12)
            self.assertEqual(loaded.hunt_mode, "walk")
            self.assertEqual(loaded.mob_custom_settings["horn"].kiting_tick_s, 0.5)
            self.assertEqual(loaded.mob_custom_settings["horn"].kite_distance_cells, 7)
            self.assertEqual(loaded.mob_custom_settings["horn"].debuff_button, "r")
            self.assertEqual(loaded.mob_custom_settings["horn"].heal_button, "q")
            self.assertEqual(loaded.mob_custom_settings["horn"].buff1_delay_s, 10)

    def test_unset_kite_distance_round_trips_as_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.ini"
            config = AppConfig(config_path=path)
            config.mob_custom_settings["horn"] = MobCustomSettings(kiting_tick_s=1.0)
            config.save()

            loaded = AppConfig(config_path=path).load()
            self.assertIsNone(loaded.mob_custom_settings["horn"].kite_distance_cells)
            self.assertNotIn("kiteDistanceCells", path.read_text(encoding="utf-8"))

    def test_invalid_kite_distances_disable_only_kiting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.ini"
            path.write_text(
                "[MonsterSettings]\n"
                "CustomBehaviors={\"zero\":{\"kiteDistanceCells\":0,\"debuffKey\":\"r\"},"
                "\"negative\":{\"kiteDistanceCells\":-2},"
                "\"blank\":{\"kiteDistanceCells\":\" \"},"
                "\"text\":{\"kiteDistanceCells\":\"far\",\"healKey\":\"q\"}}\n",
                encoding="utf-8",
            )

            loaded = AppConfig(config_path=path).load()
            self.assertIsNone(loaded.mob_custom_settings["zero"].kite_distance_cells)
            self.assertIsNone(loaded.mob_custom_settings["negative"].kite_distance_cells)
            self.assertIsNone(loaded.mob_custom_settings["blank"].kite_distance_cells)
            self.assertIsNone(loaded.mob_custom_settings["text"].kite_distance_cells)
            self.assertEqual(loaded.mob_custom_settings["zero"].debuff_button, "r")
            self.assertEqual(loaded.mob_custom_settings["text"].heal_button, "q")

    def test_removed_heal_skill_setting_is_migrated_on_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.ini"
            path.write_text(
                "[Keybindings]\nHPButton=f2\nHealSkill=1\n",
                encoding="utf-8",
            )

            config = AppConfig(config_path=path).load()
            self.assertEqual(config.hp_button, "f2")
            config.save()

            parser = configparser.ConfigParser()
            parser.read(path, encoding="utf-8")
            self.assertEqual(parser["Keybindings"]["HPButton"], "f2")
            self.assertNotIn("HealSkill", parser["Keybindings"])

    def test_client_profiles_exist(self) -> None:
        profiles = list_client_profiles(PROJECT_ROOT)
        self.assertIn("Generic", profiles)

    def test_memory_reading_follows_profile(self) -> None:
        self.assertFalse(memory_reading_enabled("Generic"))
        self.assertTrue(memory_reading_enabled("HoneyRO"))
        self.assertTrue(memory_reading_enabled("Revenant"))

    def test_revenant_profile_has_memory_addresses(self) -> None:
        from pybot.config.clients import client_supports_memory

        self.assertIn("Revenant", list_client_profiles(PROJECT_ROOT))
        self.assertTrue(client_supports_memory("Revenant", PROJECT_ROOT))

    def test_search_range_validation_is_strict(self) -> None:
        for value in (0, 8, 17, -1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Search range"):
                    hunt_runtime_config_from_settings(
                        AppSettings(search_range=value)
                    )

    def test_valid_search_range_is_preserved(self) -> None:
        config = hunt_runtime_config_from_settings(AppSettings(search_range=9))
        self.assertEqual(config.search_range_cells, 9)


class MobCatalogTests(unittest.TestCase):
    def test_load_catalog(self) -> None:
        catalog = load_mob_catalog()
        self.assertGreater(len(catalog), 0)

    def test_display_name(self) -> None:
        self.assertEqual(mob_display_name("horn"), "Horn")

    def test_resolve_mob_name_uses_catalog(self) -> None:
        parser = configparser.ConfigParser()
        parser["MonsterSettings"] = {"SelectedMonster": "1"}
        name = resolve_mob_name(parser, None)
        self.assertTrue(name)


if __name__ == "__main__":
    unittest.main()
