"""Shared pytest fixtures for the ViiperHexBots test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from pybot.paths import PROJECT_ROOT, RECOGNITION_DIR, RECOGNITION_FIXTURES_DIR


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def recognition_dir() -> Path:
    return RECOGNITION_DIR


@pytest.fixture(scope="session")
def recognition_fixtures_dir() -> Path:
    return RECOGNITION_FIXTURES_DIR
