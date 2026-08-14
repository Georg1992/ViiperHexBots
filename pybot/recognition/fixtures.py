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
    use_sprite_grf: bool = False
    recognition_regression: bool = True

    @property
    def image_dir(self) -> Path:
        return SCREENSHOTS_DIR / self.folder

    def manifest(self) -> dict:
        path = self.image_dir / "manifest.json"
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _manifest_expected_counts(self) -> dict[str, int]:
        """Per-file in-hunt-range expected counts (override the filename count).

        The filename's leading digit is the ground-truth mob count visible in
        the full screenshot. The manifest may pin a lower ``expectedCounts``
        entry for files whose mobs sit outside the current production search
        box (16 cells × CELL_SIZE_PX); entries are exact detector calibrations,
        never approximations.
        """
        raw = self.manifest().get("expectedCounts") or {}
        return {str(name): int(count) for name, count in raw.items()}

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
        expected_counts = self._manifest_expected_counts()
        fixtures: list[MobFixtureImage] = []
        for path in sorted(self.image_dir.glob("*.png")):
            match = self.pattern.match(path.name)
            if match is None:
                continue
            fixtures.append(
                MobFixtureImage(
                    file_name=path.name,
                    path=path,
                    expected_count=expected_counts.get(
                        path.name, int(match.group(1))
                    ),
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
    # ModifiedSprite captures are rendered from sprite.grf and must use the
    # static GRF descriptor, not the normal purple source-sprite descriptor.
    MobFixtureSuite(
        folder="TharaFrog",
        mob_name="thara_frog",
        pattern=re.compile(
            r"^(\d+)Tharas?_Gray_ModifiedSprite\.png$", re.IGNORECASE
        ),
        expected_fixture_count=3,
        expected_normal_count=0,
        expected_gray_count=3,
        use_sprite_grf=True,
        # These captures were taken from the pre-fix archive, whose generated
        # static SPR mapped most bright red pixels to an opaque near-black
        # palette entry. Keep them for fixture-count and forensic coverage;
        # fresh captures are required for an acceptance regression after the
        # corrected GRF is installed.
        recognition_regression=False,
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
    # Anubis — the ModifiedSprite in-world screenshots were made for legacy
    # sprites and removed; GRF descriptor-mode tests live in
    # tests/recognition/test_grf_detector_mode.py.
    # MobFixtureSuite.from_manifest(
    #     folder="Anubis",
    #     mob_name="anubis",
    #     pattern=re.compile(r"^(\d+)Anubis(?:_Gray\d*)?\.png$", re.IGNORECASE),
    # ),
    # Creamy — no SPR/ACT assets yet; fixtures kept for future use.
    # MobFixtureSuite.from_manifest(
    #     folder="Creamy",
    #     mob_name="creamy",
    #     pattern=re.compile(r"^(\d+)Creamy(?:_Gray\d*)?\.png$", re.IGNORECASE),
    # ),
    MobFixtureSuite.from_manifest(
        folder="Wolf",
        mob_name="desert_wolf",
        pattern=re.compile(r"^(\d+)Wolf(?:_Gray\d*)?\.png$", re.IGNORECASE),
    ),
    MobFixtureSuite.from_manifest(
        folder="WildRose",
        mob_name="wild_rose",
        pattern=re.compile(
            r"^(\d+)WildRose(?:_Gray\d*|False\d*)?\.png$",
            re.IGNORECASE,
        ),
    ),
)


def shipped_mob_spr_stems() -> tuple[str, ...]:
    """SPR stems for every mob under assets/mobs (one per folder)."""
    mobs_dir = PROJECT_ROOT / "assets" / "mobs"
    if not mobs_dir.is_dir():
        return ()
    stems: list[str] = []
    for folder in sorted(mobs_dir.iterdir()):
        if not folder.is_dir():
            continue
        sprite_dir = folder / "sprite"
        if not sprite_dir.is_dir():
            continue
        spr_files = sorted(sprite_dir.glob("*.spr"))
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
