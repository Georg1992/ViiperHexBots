"""Backward-compatible re-exports — use :mod:`pybot.config.runtime` instead."""

from pybot.config.runtime import (
    HuntRuntimeConfig,
    hunt_runtime_config_from_settings,
    load_runtime_config,
    resolve_mob_name,
)
from pybot.paths import CONFIG_PATH, PROJECT_ROOT

DEFAULT_CONFIG_PATH = CONFIG_PATH

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "HuntRuntimeConfig",
    "PROJECT_ROOT",
    "hunt_runtime_config_from_settings",
    "load_runtime_config",
    "resolve_mob_name",
]
