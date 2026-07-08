"""Shared project paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSETS_DIR = PROJECT_ROOT / "assets"
MOBS_DIR = ASSETS_DIR / "mobs"
DESCRIPTORS_DIR = ASSETS_DIR / "generated_descriptors"
MODIFIED_DESCRIPTORS_DIR = DESCRIPTORS_DIR / "modified"
MODIFIED_MOBS_DIR = ASSETS_DIR / "modified_mobs"

CLIENTS_DIR = PROJECT_ROOT / "clients"
LOGS_DIR = PROJECT_ROOT / "logs"
SESSIONS_DIR = LOGS_DIR / "sessions"

MOB_RECOGNITION_DIR = PROJECT_ROOT / "pybot" / "recognition"
RECOGNITION_DIR = MOB_RECOGNITION_DIR
RECOGNITION_FIXTURES_DIR = RECOGNITION_DIR / "test-fixtures"

VIIPER_EXE = PROJECT_ROOT / "VIIPER" / "dist" / "viiper.exe"
CONFIG_PATH = PROJECT_ROOT / "config.ini"
