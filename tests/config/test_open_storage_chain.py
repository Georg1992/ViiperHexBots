"""Open Storage keychain load/save and migration."""

from __future__ import annotations

import configparser
from pathlib import Path

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


def test_arrow_keysyms_and_scan_codes() -> None:
    assert keysym_to_key_name("Down") == "down"
    assert keysym_to_key_name("Up") == "up"
    assert key_name_to_scan_code("down") > 0
    assert key_name_to_scan_code("up") > 0
    assert key_name_to_scan_code("left") > 0
    assert key_name_to_scan_code("right") > 0


def test_migrate_legacy_open_storage_button() -> None:
    parser = configparser.ConfigParser()
    parser["Keybindings"] = {"OpenStorageButton": "f8"}
    steps = _load_open_storage_chain(parser)
    assert len(steps) == 1
    assert steps[0].button == "f8"
    assert steps[0].delay_ms == 0


def test_load_open_storage_chain_json() -> None:
    parser = configparser.ConfigParser()
    parser["Keybindings"] = {
        "OpenStorageChain": '[{"key":"f8","delay":100},{"key":"down","delay":100},'
        '{"key":"enter","delay":0}]'
    }
    steps = _load_open_storage_chain(parser)
    assert [(s.button, s.delay_ms) for s in steps] == [
        ("f8", 100),
        ("down", 100),
        ("enter", 0),
    ]


def test_save_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.ini"
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
    assert [(s.button, s.delay_ms) for s in loaded.open_storage_chain] == [
        ("f8", 100),
        ("down", 100),
        ("enter", 0),
    ]
    text = path.read_text(encoding="utf-8").lower()
    assert "openstoragechain" in text
    assert "openstoragebutton" not in text


def test_runtime_steps_from_settings() -> None:
    settings = AppSettings(
        open_storage_chain=[
            KeyChainStep("f8", 100),
            KeyChainStep("down", 50),
            KeyChainStep("", 0),
        ]
    )
    cfg = hunt_runtime_config_from_settings(settings)
    assert len(cfg.open_storage_steps) == 2
    assert cfg.open_storage_steps[0][0] == "f8"
    assert cfg.open_storage_steps[0][2] == 100
    assert cfg.open_storage_steps[1][0] == "down"
    assert cfg.open_storage_steps[0][1] == key_name_to_scan_code("f8")


def test_play_key_chain_order(monkeypatch) -> None:
    backend = ShadowInputBackend()
    events: list[tuple] = []

    def fake_tap(scan_code: int, *, press_s: float = 0.05, after_s: float = 0.30) -> bool:
        events.append(("tap", scan_code))
        return True

    def fake_sleep(seconds: float) -> None:
        events.append(("sleep", seconds))

    monkeypatch.setattr(backend, "key_tap", fake_tap)
    monkeypatch.setattr(
        "pybot.runtime.input.input_backend.time.sleep", fake_sleep
    )

    steps = (
        ("f8", 66, 100),
        ("down", 80, 50),
        ("enter", 28, 0),
    )
    assert backend.play_key_chain(steps) is True
    assert events == [
        ("tap", 66),
        ("sleep", 0.1),
        ("tap", 80),
        ("sleep", 0.05),
        ("tap", 28),
    ]


def test_save_open_storage_chain_json() -> None:
    raw = _save_open_storage_chain(
        [KeyChainStep("f8", 100), KeyChainStep("", 0), KeyChainStep("enter", 0)]
    )
    assert raw == '[{"key":"f8","delay":100},{"key":"enter","delay":0}]'
