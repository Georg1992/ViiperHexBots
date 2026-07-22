"""Import SPR+ACT pairs into assets/mobs and build descriptors."""

from __future__ import annotations

import shutil
from pathlib import Path

from pybot.mobs.catalog import MobEntry, descriptor_path, mob_display_name
from pybot.paths import MOBS_DIR, PROJECT_ROOT
from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.descriptors.descriptor_builder import DescriptorBuilder


class MobImportError(ValueError):
    """Raised when dropped/browsed paths cannot form a valid SPR+ACT pair."""


def resolve_spr_act_paths(paths: list[Path]) -> tuple[Path, Path]:
    """Resolve a matching ``.spr`` + ``.act`` pair from dropped/browsed paths.

    Accepts:
    - two files with the same stem (``.spr`` and ``.act``), or
    - one directory containing exactly one such pair (or a clear stem match).
    """
    if not paths:
        raise MobImportError("no files provided")
    resolved = [Path(p).expanduser().resolve() for p in paths]

    if len(resolved) == 1 and resolved[0].is_dir():
        return _pair_from_directory(resolved[0])

    files = []
    for path in resolved:
        if path.is_dir():
            raise MobImportError(
                "drop either one folder or the .spr and .act files, not both"
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


def _pair_from_directory(folder: Path) -> tuple[Path, Path]:
    spr_files = sorted(folder.glob("*.spr")) + sorted(folder.glob("*.SPR"))
    # glob is case-sensitive on some platforms; dedupe by resolved path.
    seen: set[Path] = set()
    unique_spr: list[Path] = []
    for path in spr_files:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique_spr.append(path)
    if not unique_spr:
        raise MobImportError(f"no .spr file in folder: {folder}")
    if len(unique_spr) > 1:
        # Prefer a file whose stem matches the folder name.
        folder_key = folder.name.lower()
        matched = [p for p in unique_spr if p.stem.lower() == folder_key]
        if len(matched) == 1:
            unique_spr = matched
        else:
            raise MobImportError(
                f"multiple .spr files in folder: {folder} "
                f"({', '.join(p.name for p in unique_spr)})"
            )
    spr = unique_spr[0]
    act = spr.with_suffix(".act")
    if not act.is_file():
        act_upper = spr.with_suffix(".ACT")
        if act_upper.is_file():
            act = act_upper
        else:
            raise MobImportError(f"missing matching ACT for {spr.name} in {folder}")
    return spr, act


def mob_assets_exist(stem: str) -> bool:
    """True when ``assets/mobs/{stem}/{stem}.spr`` already exists."""
    key = stem.lower()
    spr = MOBS_DIR / key / f"{key}.spr"
    return spr.is_file()


def install_mob_assets(
    spr: Path,
    act: Path,
    *,
    overwrite: bool = False,
) -> str:
    """Copy SPR/ACT into ``assets/mobs/{stem}/`` and return the lowercase stem."""
    stem = spr.stem.lower()
    if act.stem.lower() != stem:
        raise MobImportError(
            f"SPR/ACT stems must match ({spr.name} vs {act.name})"
        )
    dest_dir = MOBS_DIR / stem
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
    descriptor = DescriptorBuilder(PROJECT_ROOT).build(key, force=True)
    path = descriptor_path(key)
    if not path.is_file():
        raise RuntimeError(f"descriptor missing after build: {path}")
    return descriptor


def import_mob_from_paths(
    paths: list[Path],
    *,
    overwrite: bool = False,
) -> MobEntry:
    """Resolve paths, install assets, build descriptor, return catalog entry."""
    spr, act = resolve_spr_act_paths(paths)
    stem = install_mob_assets(spr, act, overwrite=overwrite)
    build_mob_descriptor(stem)
    return MobEntry(
        asset_name=stem,
        display_name=mob_display_name(stem),
        descriptor_name=stem,
    )
