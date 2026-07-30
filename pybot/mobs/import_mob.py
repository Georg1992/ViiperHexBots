"""Import SPR+ACT pairs into assets/mobs and build descriptors."""

from __future__ import annotations

import shutil
from pathlib import Path

from pybot.mobs.catalog import MobEntry, descriptor_path, mob_display_name
from pybot.paths import MOBS_DIR, PROJECT_ROOT
from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.descriptors.descriptor_builder import DescriptorBuilder


class MobImportError(ValueError):
    """Raised when browsed paths cannot form a valid SPR+ACT pair."""


def resolve_spr_act_paths(paths: list[Path]) -> tuple[Path, Path]:
    """Resolve a matching ``.spr`` + ``.act`` pair from browsed paths.

    Requires exactly one ``.spr`` file and one ``.act`` file with the
    same stem.
    """
    if not paths:
        raise MobImportError("no files provided")
    resolved = [Path(p).expanduser().resolve() for p in paths]

    files: list[Path] = []
    for path in resolved:
        if path.is_dir():
            raise MobImportError(
                "folders are not supported — select the .spr and .act files directly"
            )
        if not path.is_file():
            raise MobImportError(f"path not found: {path}")
        files.append(path)

    spr_files = [p for p in files if p.suffix.lower() == ".spr"]
    act_files = [p for p in files if p.suffix.lower() == ".act"]
    other = [
        p for p in files
        if p.suffix.lower() not in {".spr", ".act"}
    ]
    if other:
        raise MobImportError(
            f"unsupported file type(s): {', '.join(p.name for p in other)}"
        )
    if len(spr_files) != 1 or len(act_files) != 1:
        raise MobImportError("need exactly one .spr and one .act file")
    spr = spr_files[0]
    act = act_files[0]
    if spr.stem.lower() != act.stem.lower():
        raise MobImportError(
            f"SPR/ACT stems must match ({spr.name} vs {act.name})"
        )
    return spr, act


def mob_assets_exist(stem: str) -> bool:
    """True when ``assets/mobs/{stem}/sprite/{stem}.spr`` already exists."""
    key = stem.lower()
    spr = MOBS_DIR / key / "sprite" / f"{key}.spr"
    return spr.is_file()


def install_mob_assets(
    spr: Path,
    act: Path,
    *,
    overwrite: bool = False,
) -> str:
    """Copy SPR/ACT into ``assets/mobs/{stem}/sprite/`` and return the lowercase stem."""
    stem = spr.stem.lower()
    if act.stem.lower() != stem:
        raise MobImportError(
            f"SPR/ACT stems must match ({spr.name} vs {act.name})"
        )
    dest_dir = MOBS_DIR / stem / "sprite"
    dest_spr = dest_dir / f"{stem}.spr"
    dest_act = dest_dir / f"{stem}.act"
    if dest_spr.is_file() and not overwrite:
        raise MobImportError(
            f"mob '{stem}' already exists — pass overwrite=True to replace"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(spr, dest_spr)
    shutil.copy2(act, dest_act)
    return stem


def build_mob_descriptor(stem: str) -> MobDescriptor:
    """Force-build the descriptor for an installed mob stem."""
    key = stem.lower()
    builder = DescriptorBuilder(PROJECT_ROOT)
    descriptor = builder.build(key, force=True)
    path = descriptor_path(key)
    if not path.is_file():
        raise RuntimeError(f"descriptor missing after build: {path}")
    return descriptor


def build_modified_sprite_descriptor(stem: str) -> MobDescriptor | None:
    """Force-build the modified (big+red) sprite descriptor (best-effort)."""
    key = stem.lower()
    builder = DescriptorBuilder(PROJECT_ROOT)
    return builder.build_modified_sprite(key, force=True)


def import_mob_from_paths(
    paths: list[Path],
    *,
    overwrite: bool = False,
) -> MobEntry:
    """Resolve paths, install assets, build descriptor, return catalog entry."""
    spr, act = resolve_spr_act_paths(paths)
    stem = install_mob_assets(spr, act, overwrite=overwrite)
    build_mob_descriptor(stem)
    try:
        build_modified_sprite_descriptor(stem)
    except Exception as exc:
        # Best-effort: normal descriptor is already built.
        print(
            f"[IMPORT] modified-sprite descriptor skipped for '{stem}': {exc}"
        )
    return MobEntry(
        asset_name=stem,
        display_name=mob_display_name(stem),
        descriptor_name=stem,
    )
