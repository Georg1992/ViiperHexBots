"""Mob screenshot fixture discovery for recognition regression tests."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pybot.paths import PROJECT_ROOT, RECOGNITION_FIXTURES_DIR
from pybot.runtime.capture.window_roi import crop_frame_to_hunt_search_roi
from pybot.runtime.constants import CELL_SIZE_PX, DEFAULT_SEARCH_RANGE_CELLS

SCREENSHOTS_DIR = RECOGNITION_FIXTURES_DIR / "game-screenshots"
MAX_SEARCH_RANGE_CELLS = DEFAULT_SEARCH_RANGE_CELLS


@dataclass(frozen=True)
class MobFixtureImage:
    file_name: str
    path: Path
    expected_count: int
    gray_world: bool


@dataclass(frozen=True)
class MobFixtureSuite:
    folder: str
    mob_name: str
    pattern: re.Pattern[str]
    expected_fixture_count: int = 8
    expected_normal_count: int = 4
    expected_gray_count: int = 4

    @property
    def image_dir(self) -> Path:
        return SCREENSHOTS_DIR / self.folder

    def manifest(self) -> dict:
        path = self.image_dir / "manifest.json"
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @classmethod
    def from_manifest(
        cls,
        *,
        folder: str,
        mob_name: str,
        pattern: re.Pattern[str],
    ) -> "MobFixtureSuite":
        suite = cls(folder=folder, mob_name=mob_name, pattern=pattern)
        manifest = suite.manifest()
        return cls(
            folder=folder,
            mob_name=mob_name,
            pattern=pattern,
            expected_fixture_count=int(manifest.get("fixtureCount", 8)),
            expected_normal_count=int(manifest.get("normalFixtureCount", 4)),
            expected_gray_count=int(manifest.get("grayFixtureCount", 4)),
        )

    def images(self) -> list[MobFixtureImage]:
        if not self.image_dir.is_dir():
            return []
        fixtures: list[MobFixtureImage] = []
        for path in sorted(self.image_dir.glob("*.png")):
            match = self.pattern.match(path.name)
            if match is None:
                continue
            fixtures.append(
                MobFixtureImage(
                    file_name=path.name,
                    path=path,
                    expected_count=int(match.group(1)),
                    gray_world="_Gray" in path.stem,
                )
            )
        return fixtures


MOB_FIXTURE_SUITES: tuple[MobFixtureSuite, ...] = (
    MobFixtureSuite.from_manifest(
        folder="Horn",
        mob_name="horn",
        pattern=re.compile(r"^(\d+)Horn(?:_Gray\d*)?\.png$", re.IGNORECASE),
    ),
    MobFixtureSuite.from_manifest(
        folder="TharaFrog",
        mob_name="thara_frog",
        pattern=re.compile(r"^(\d+)Tharas?(?:_Gray)?\.png$", re.IGNORECASE),
    ),
    MobFixtureSuite.from_manifest(
        folder="Alligator",
        mob_name="alligator",
        pattern=re.compile(r"^(\d+)Alligator(?:_Gray)?\.png$", re.IGNORECASE),
    ),
    MobFixtureSuite.from_manifest(
        folder="Noxious",
        mob_name="noxious",
        pattern=re.compile(r"^(\d+)Noxious(?:_Gray)?\.png$", re.IGNORECASE),
    ),
)


def suite_by_folder(folder: str) -> MobFixtureSuite | None:
    for suite in MOB_FIXTURE_SUITES:
        if suite.folder == folder:
            return suite
    return None


def shipped_mob_spr_stems() -> tuple[str, ...]:
    """SPR stems for every mob under assets/mobs (one per folder)."""
    mobs_dir = PROJECT_ROOT / "assets" / "mobs"
    if not mobs_dir.is_dir():
        return ()
    stems: list[str] = []
    for folder in sorted(mobs_dir.iterdir()):
        if not folder.is_dir():
            continue
        spr_files = sorted(folder.glob("*.spr"))
        if spr_files:
            stems.append(spr_files[0].stem.lower())
    return tuple(stems)


def default_horn_fixture() -> Path:
    """Representative horn screenshot used by tracker/state integration tests."""
    path = SCREENSHOTS_DIR / "Horn" / "3Horn.png"
    if not path.is_file():
        raise FileNotFoundError(f"missing default horn fixture: {path}")
    return path


def fixture_search_frame(
    frame: np.ndarray,
    *,
    search_range_cells: int = MAX_SEARCH_RANGE_CELLS,
    cell_size_px: int = CELL_SIZE_PX,
) -> np.ndarray:
    """Crop a fixture screenshot to the max GUI hunt search range (production parity)."""
    return crop_frame_to_hunt_search_roi(
        frame,
        search_range_cells=search_range_cells,
        cell_size_px=cell_size_px,
    )
