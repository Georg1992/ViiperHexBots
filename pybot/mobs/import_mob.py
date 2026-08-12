"""Import SPR+ACT pairs into assets/mobs and build descriptors."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from pybot.mobs.catalog import MobEntry, descriptor_path, mob_display_name
from pybot.paths import DESCRIPTORS_DIR, MOBS_DIR, PROJECT_ROOT
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


def delete_mob_assets(asset_name: str, descriptor_name: str) -> None:
    """Delete one mob's source assets, descriptors, and GRF entries."""
    asset_key = asset_name.strip()
    descriptor_key = descriptor_name.strip().lower()
    if not asset_key or not descriptor_key:
        raise MobImportError("mob name cannot be empty")

    mob_dir = (MOBS_DIR / asset_key).resolve()
    mobs_root = MOBS_DIR.resolve()
    descriptor_dir = (DESCRIPTORS_DIR / descriptor_key).resolve()
    descriptors_root = DESCRIPTORS_DIR.resolve()
    if mobs_root not in mob_dir.parents or descriptors_root not in descriptor_dir.parents:
        raise MobImportError("refusing to delete a path outside the mob asset roots")
    if not mob_dir.is_dir():
        raise MobImportError(f"mob assets not found: {asset_name}")

    # Stage both directories out of the catalog before regenerating the
    # archive. This makes the archive and filesystem change as one operation:
    # a GRF or filesystem failure restores the staged directories and archive.
    from pybot.mobs.sprite_grf import sync_sprite_grf

    archive_path = PROJECT_ROOT / "sprite.grf"
    archive_backup = archive_path.read_bytes() if archive_path.is_file() else None
    stage_root = Path(tempfile.mkdtemp(prefix=".mob-delete-", dir=str(PROJECT_ROOT)))
    staged_mob_dir = stage_root / "mob"
    staged_descriptor_dir = stage_root / "descriptor"
    try:
        mob_dir.rename(staged_mob_dir)
        if descriptor_dir.is_dir():
            descriptor_dir.rename(staged_descriptor_dir)
        sync_sprite_grf(PROJECT_ROOT, remove_mob_name=descriptor_key)
        shutil.rmtree(stage_root)
    except Exception as exc:
        rollback_errors: list[Exception] = []
        for staged, original in (
            (staged_mob_dir, mob_dir),
            (staged_descriptor_dir, descriptor_dir),
        ):
            try:
                if staged.exists() and not original.exists():
                    staged.rename(original)
            except Exception as rollback_exc:
                rollback_errors.append(rollback_exc)
        try:
            shutil.rmtree(stage_root, ignore_errors=False)
        except Exception as rollback_exc:
            rollback_errors.append(rollback_exc)
        try:
            if archive_backup is None:
                archive_path.unlink(missing_ok=True)
            else:
                archive_path.write_bytes(archive_backup)
        except Exception as rollback_exc:
            rollback_errors.append(rollback_exc)
        if rollback_errors:
            details = "; ".join(str(error) for error in rollback_errors)
            raise RuntimeError(
                f"mob deletion failed and rollback failed: {details}"
            ) from exc
        raise


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
    stem = spr.stem.lower()
    if mob_assets_exist(stem) and not overwrite:
        raise MobImportError(
            f"mob '{stem}' already exists — pass overwrite=True to replace"
        )
    asset_dir = MOBS_DIR / stem
    descriptor_dir = DESCRIPTORS_DIR / stem
    archive_path = PROJECT_ROOT / "sprite.grf"
    archive_backup = archive_path.read_bytes() if archive_path.is_file() else None
    stage_root = Path(tempfile.mkdtemp(prefix=".mob-import-", dir=str(PROJECT_ROOT)))
    staged_asset_dir = stage_root / "mob"
    staged_descriptor_dir = stage_root / "descriptor"

    from pybot.mobs.sprite_grf import sync_sprite_grf

    try:
        # Stage existing data so overwrite and descriptor rebuild failures can
        # restore the complete pre-import state.
        if asset_dir.is_dir():
            asset_dir.rename(staged_asset_dir)
        if descriptor_dir.is_dir():
            descriptor_dir.rename(staged_descriptor_dir)

        stem = install_mob_assets(spr, act, overwrite=False)
        build_mob_descriptor(stem)
        try:
            build_modified_sprite_descriptor(stem)
        except Exception as exc:
            # Best-effort: normal descriptor is already built.
            print(
                f"[IMPORT] modified-sprite descriptor skipped for '{stem}': {exc}"
            )

        # Keep the archive in sync immediately; waiting for the next
        # application startup would leave the newly added mob unavailable in
        # GRF mode.
        sync_sprite_grf(PROJECT_ROOT)
        shutil.rmtree(stage_root)
    except Exception as exc:
        rollback_errors: list[Exception] = []
        for path in (asset_dir, descriptor_dir):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
            except Exception as rollback_exc:
                rollback_errors.append(rollback_exc)
        for staged, original in (
            (staged_asset_dir, asset_dir),
            (staged_descriptor_dir, descriptor_dir),
        ):
            try:
                if staged.exists() and not original.exists():
                    staged.rename(original)
            except Exception as rollback_exc:
                rollback_errors.append(rollback_exc)
        try:
            shutil.rmtree(stage_root, ignore_errors=False)
        except Exception as rollback_exc:
            rollback_errors.append(rollback_exc)
        try:
            if archive_backup is None:
                archive_path.unlink(missing_ok=True)
            else:
                archive_path.write_bytes(archive_backup)
        except Exception as rollback_exc:
            rollback_errors.append(rollback_exc)
        if rollback_errors:
            details = "; ".join(str(error) for error in rollback_errors)
            raise RuntimeError(
                f"mob import failed and rollback failed: {details}"
            ) from exc
        raise

    return MobEntry(
        asset_name=stem,
        display_name=mob_display_name(stem),
        descriptor_name=stem,
    )
