"""Shared project paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSETS_DIR = PROJECT_ROOT / "assets"
MOBS_DIR = ASSETS_DIR / "mobs"
DESCRIPTORS_DIR = ASSETS_DIR / "generated_descriptors"

LOGS_DIR = PROJECT_ROOT / "logs"
SESSIONS_DIR = LOGS_DIR / "sessions"

RECOGNITION_DIR = PROJECT_ROOT / "pybot" / "recognition"
TEST_FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
RECOGNITION_FIXTURES_DIR = TEST_FIXTURES_DIR / "recognition"

CONFIG_PATH = PROJECT_ROOT / "config.ini"
