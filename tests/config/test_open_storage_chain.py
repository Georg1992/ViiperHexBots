"""Open Storage keychain load/save and migration."""

from __future__ import annotations

import configparser
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pybot.config.ini_store import (
    _load_open_storage_chain,
    _save_open_storage_chain,
    load_settings,
    save_settings,
)
from pybot.config.runtime import hunt_runtime_config_from_settings
from pybot.config.schema import AppSettings, KeyChainStep
from pybot.runtime.input.scan_codes import key_name_to_scan_code, keysym_to_key_name
from pybot.runtime.input.input_backend import ShadowInputBackend


class OpenStorageChainTests(unittest.TestCase):
    def test_arrow_keysyms_and_scan_codes(self) -> None:
        self.assertEqual(keysym_to_key_name("Down"), "down")
        self.assertEqual(keysym_to_key_name("Up"), "up")
        self.assertGreater(key_name_to_scan_code("down"), 0)
        self.assertGreater(key_name_to_scan_code("up"), 0)
        self.assertGreater(key_name_to_scan_code("left"), 0)
        self.assertGreater(key_name_to_scan_code("right"), 0)

    def test_migrate_legacy_open_storage_button(self) -> None:
        parser = configparser.ConfigParser()
        parser["Keybindings"] = {"OpenStorageButton": "f8"}
        steps = _load_open_storage_chain(parser)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].button, "f8")
        self.assertEqual(steps[0].delay_ms, 0)

    def test_load_open_storage_chain_json(self) -> None:
        parser = configparser.ConfigParser()
        parser["Keybindings"] = {
            "OpenStorageChain": '[{"key":"f8","delay":100},{"key":"down","delay":100},'
            '{"key":"enter","delay":0}]'
        }
        steps = _load_open_storage_chain(parser)
        self.assertEqual(
            [(s.button, s.delay_ms) for s in steps],
            [("f8", 100), ("down", 100), ("enter", 0)],
        )

    def test_save_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.ini"
            settings = AppSettings(
                config_path=path,
                open_storage_chain=[
                    KeyChainStep("f8", 100),
                    KeyChainStep("down", 100),
                    KeyChainStep("enter", 0),
                ],
            )
            save_settings(settings)
            loaded = load_settings(path)
            self.assertEqual(
                [(s.button, s.delay_ms) for s in loaded.open_storage_chain],
                [("f8", 100), ("down", 100), ("enter", 0)],
            )
            text = path.read_text(encoding="utf-8").lower()
            self.assertIn("openstoragechain", text)
            self.assertNotIn("openstoragebutton", text)

    def test_runtime_steps_from_settings(self) -> None:
        settings = AppSettings(
            open_storage_chain=[
                KeyChainStep("f8", 100),
                KeyChainStep("down", 50),
                KeyChainStep("", 0),
            ]
        )
        cfg = hunt_runtime_config_from_settings(settings)
        self.assertEqual(len(cfg.open_storage_steps), 2)
        self.assertEqual(cfg.open_storage_steps[0][0], "f8")
        self.assertEqual(cfg.open_storage_steps[0][2], 100)
        self.assertEqual(cfg.open_storage_steps[1][0], "down")
        self.assertEqual(cfg.open_storage_steps[0][1], key_name_to_scan_code("f8"))

    def test_play_key_chain_order(self) -> None:
        backend = ShadowInputBackend()
        events: list[tuple] = []

        def fake_tap(scan_code: int, *, press_s: float = 0.05, after_s: float = 0.30) -> bool:
            events.append(("tap", scan_code))
            return True

        def fake_sleep(seconds: float) -> None:
            events.append(("sleep", seconds))

        with (
            patch.object(backend, "key_tap", fake_tap),
            patch("pybot.runtime.input.input_backend.time.sleep", fake_sleep),
        ):
            steps = (
                ("f8", 66, 100),
                ("down", 80, 50),
                ("enter", 28, 0),
            )
            self.assertTrue(backend.play_key_chain(steps))
            self.assertEqual(
                events,
                [("tap", 66), ("sleep", 0.1), ("tap", 80), ("sleep", 0.05), ("tap", 28)],
            )

    def test_save_open_storage_chain_json(self) -> None:
        raw = _save_open_storage_chain(
            [KeyChainStep("f8", 100), KeyChainStep("", 0), KeyChainStep("enter", 0)]
        )
        self.assertEqual(raw, '[{"key":"f8","delay":100},{"key":"enter","delay":0}]')


if __name__ == "__main__":
    unittest.main()
