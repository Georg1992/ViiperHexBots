"""Mob descriptor catalog from assets/generated_descriptors/."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pybot.mobs import act_transform
from pybot.paths import (
    DESCRIPTORS_DIR,
    MODIFIED_DESCRIPTORS_DIR,
    MOBS_DIR,
    MODIFIED_MOBS_DIR,
    PROJECT_ROOT,
)


@dataclass(frozen=True)
class MobEntry:
    asset_name: str
    display_name: str
    descriptor_name: str


def mob_display_name(asset_name: str) -> str:
    display = asset_name.replace("_", " ").replace("-", " ")
    if not display:
        return asset_name
    return display[0].upper() + display[1:]


def _scan_asset_pairs() -> list[tuple[str, str]]:
    if not MOBS_DIR.is_dir():
        return []
    pairs: list[tuple[str, str]] = []
    for mob_dir in sorted(MOBS_DIR.iterdir()):
        if not mob_dir.is_dir():
            continue
        for spr_path in sorted(mob_dir.glob("*.spr")):
            spr_stem = spr_path.stem
            act_path = mob_dir / f"{spr_stem}.act"
            if act_path.is_file():
                pairs.append((mob_dir.name, spr_stem))
                break
    return pairs


def descriptor_path(spr_stem: str, *, modified: bool = False) -> Path:
    stem = spr_stem.lower()
    if modified:
        return MODIFIED_DESCRIPTORS_DIR / stem / "descriptor.json"
    return DESCRIPTORS_DIR / stem / "descriptor.json"


def _build_descriptor(asset_name: str, spr_stem: str, _logger) -> None:
    descriptor_path_file = descriptor_path(spr_stem, modified=False)
    if descriptor_path_file.is_file():
        return
    from pybot.recognition.detector.descriptors.descriptor_builder import DescriptorBuilder

    _logger(f"[AUTO-BUILD] {asset_name}: SPR/ACT found, building descriptor ({spr_stem})...")
    DescriptorBuilder(PROJECT_ROOT).build(spr_stem, force=True)
    _logger(f"[AUTO-BUILD] {asset_name}: descriptor ready")


def _build_modified_descriptor(asset_name: str, spr_stem: str, _logger) -> None:
    descriptor_path_file = descriptor_path(spr_stem, modified=True)
    if descriptor_path_file.is_file():
        return
    modified_spr = MODIFIED_MOBS_DIR / asset_name / f"{spr_stem}.spr"
    modified_act = MODIFIED_MOBS_DIR / asset_name / f"{spr_stem}.act"
    if not modified_spr.is_file() or not modified_act.is_file():
        return
    from pybot.recognition.detector.descriptors.descriptor_builder import DescriptorBuilder

    _logger(
        f"[AUTO-BUILD] {asset_name}: building modified descriptor ({spr_stem})..."
    )
    DescriptorBuilder(PROJECT_ROOT).build_modified(
        asset_name,
        spr_stem,
        force=True,
    )
    _logger(f"[AUTO-BUILD] {asset_name}: modified descriptor ready")


def _build_modified_mob(asset_name: str, spr_stem: str, _logger) -> None:
    target_dir = MODIFIED_MOBS_DIR / asset_name
    target_act = target_dir / f"{spr_stem}.act"
    target_spr = target_dir / f"{spr_stem}.spr"
    if target_act.is_file() and target_spr.is_file():
        return

    src_dir = MOBS_DIR / asset_name
    target_dir.mkdir(parents=True, exist_ok=True)
    _logger(f"[MODIFY] {asset_name}: creating modified SPR/ACT ({spr_stem})...")
    shutil.copyfile(src_dir / f"{spr_stem}.spr", target_spr)
    act_transform.transform(src_dir / f"{spr_stem}.act", target_act)
    _logger(f"[MODIFY] {asset_name}: modified assets ready")


def ensure_mob_assets(*, log_fn: Callable[[str], None] | None = None) -> None:
    _logger = log_fn or print
    if not MOBS_DIR.is_dir():
        return
    for asset_name, spr_stem in _scan_asset_pairs():
        try:
            _build_descriptor(asset_name, spr_stem, _logger)
        except Exception as exc:
            _logger(f"[AUTO-BUILD] {asset_name}: build failed — {exc}")
        try:
            _build_modified_mob(asset_name, spr_stem, _logger)
        except Exception as exc:
            _logger(f"[MODIFY] {asset_name}: modify failed — {exc}")
        try:
            _build_modified_descriptor(asset_name, spr_stem, _logger)
        except Exception as exc:
            _logger(f"[AUTO-BUILD] {asset_name}: modified descriptor failed — {exc}")


def load_mob_catalog(*, ensure_assets: bool = False) -> list[MobEntry]:
    if ensure_assets:
        ensure_mob_assets()
    if not MOBS_DIR.is_dir():
        return []

    entries: list[MobEntry] = []
    for asset_name, spr_stem in _scan_asset_pairs():
        descriptor_path_file = descriptor_path(spr_stem, modified=False)
        if not descriptor_path_file.is_file():
            continue
        entries.append(
            MobEntry(
                asset_name=asset_name,
                display_name=mob_display_name(asset_name),
                descriptor_name=spr_stem,
            )
        )
    return entries


def mob_folder_by_index(catalog: list[MobEntry], index: int) -> str:
    if not catalog:
        return "horn"
    clamped = max(1, min(index, len(catalog)))
    return catalog[clamped - 1].descriptor_name


def resolve_mob_descriptor_name(
    *,
    selected_monster: int,
    mob_name: str | None = None,
) -> str:
    if mob_name:
        return mob_name
    catalog = load_mob_catalog()
    if not catalog:
        raise RuntimeError("No mob catalog found. Run build-mob-descriptor.ps1 first.")
    return mob_folder_by_index(catalog, selected_monster)
