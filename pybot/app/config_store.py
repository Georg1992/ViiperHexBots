"""Read/write config.ini for the Python application."""

from __future__ import annotations

from pybot.config.clients import client_supports_memory, list_client_profiles
from pybot.config.ini_store import load_settings, save_settings
from pybot.config.schema import AppSettings
from pybot.paths import CONFIG_PATH


class AppConfig(AppSettings):
    def load(self) -> AppConfig:
        loaded = load_settings(self.config_path)
        self.__dict__.update(loaded.__dict__)
        return self

    def save(self) -> None:
        save_settings(self)


DEFAULT_CONFIG_PATH = CONFIG_PATH

__all__ = [
    "AppConfig",
    "DEFAULT_CONFIG_PATH",
    "client_supports_memory",
    "list_client_profiles",
]
