"""Mob screenshot fixture discovery for recognition regression tests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pybot.paths import RECOGNITION_FIXTURES_DIR

SCREENSHOTS_DIR = RECOGNITION_FIXTURES_DIR / "game-screenshots"


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

    @property
    def image_dir(self) -> Path:
        return SCREENSHOTS_DIR / self.folder

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
    MobFixtureSuite(
        folder="Horn",
        mob_name="horn",
        pattern=re.compile(r"^(\d+)Horn(?:_Gray\d*)?\.png$", re.IGNORECASE),
    ),
    MobFixtureSuite(
        folder="TharaFrog",
        mob_name="thara_frog",
        pattern=re.compile(r"^(\d+)Tharas?(?:_Gray)?\.png$", re.IGNORECASE),
    ),
    MobFixtureSuite(
        folder="Alligator",
        mob_name="alligator",
        pattern=re.compile(r"^(\d+)Alligator(?:_Gray)?\.png$", re.IGNORECASE),
    ),
)


def suite_by_folder(folder: str) -> MobFixtureSuite | None:
    for suite in MOB_FIXTURE_SUITES:
        if suite.folder == folder:
            return suite
    return None


def default_horn_fixture() -> Path:
    """Representative horn screenshot used by tracker/state integration tests."""
    path = SCREENSHOTS_DIR / "Horn" / "3Horn.png"
    if not path.is_file():
        raise FileNotFoundError(f"missing default horn fixture: {path}")
    return path
