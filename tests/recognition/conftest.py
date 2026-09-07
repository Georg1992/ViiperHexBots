"""Rebuild descriptors to the current version before recognition tests."""

from __future__ import annotations

import pytest

from pybot.mobs.catalog import ensure_mob_assets


@pytest.fixture(scope="session", autouse=True)
def _ensure_current_mob_descriptors() -> None:
    ensure_mob_assets(log_fn=lambda _message: None)
