"""Application layer tests."""

from __future__ import annotations

import configparser
import tempfile
import unittest
from pathlib import Path

from pybot.app.config_store import AppConfig, list_client_profiles
from pybot.config.clients import memory_reading_enabled
from pybot.config.runtime import resolve_mob_name
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
            config.save()

            loaded = AppConfig(config_path=path).load()
            self.assertEqual(loaded.window_id, 123)
            self.assertEqual(loaded.window_title, "Test Game")
            self.assertEqual(loaded.search_range, 12)
            self.assertEqual(loaded.hunt_mode, "walk")

    def test_client_profiles_exist(self) -> None:
        profiles = list_client_profiles(PROJECT_ROOT)
        self.assertIn("Generic", profiles)

    def test_memory_reading_follows_profile(self) -> None:
        self.assertFalse(memory_reading_enabled("Generic"))
        self.assertTrue(memory_reading_enabled("HoneyRO"))


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
