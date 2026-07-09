"""Mob screenshot fixture discovery for recognition regression tests."""

from __future__ import annotations

import json
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


def default_horn_fixture() -> Path:
    """Representative horn screenshot used by tracker/state integration tests."""
    path = SCREENSHOTS_DIR / "Horn" / "3Horn.png"
    if not path.is_file():
        raise FileNotFoundError(f"missing default horn fixture: {path}")
    return path
