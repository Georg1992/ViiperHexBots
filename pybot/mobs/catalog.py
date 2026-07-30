"""Mob descriptor catalog from assets/generated_descriptors/."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pybot.paths import DESCRIPTORS_DIR, MOBS_DIR, PROJECT_ROOT
from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.descriptors.descriptor_builder import DESCRIPTOR_VERSION


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
        sprite_dir = mob_dir / "sprite"
        if not sprite_dir.is_dir():
            continue
        for spr_path in sorted(sprite_dir.glob("*.spr")):
            spr_stem = spr_path.stem
            act_path = sprite_dir / f"{spr_stem}.act"
            if act_path.is_file():
                pairs.append((mob_dir.name, spr_stem))
                break
    return pairs


def descriptor_path(spr_stem: str) -> Path:
    return DESCRIPTORS_DIR / spr_stem.lower() / "descriptor.json"


def modified_sprite_descriptor_path(spr_stem: str) -> Path:
    return DESCRIPTORS_DIR / spr_stem.lower() / "modified_sprite_descriptor.json"


def _descriptor_needs_rebuild(descriptor_path_file: Path) -> bool:
    """True when the descriptor file is missing, unreadable, or below DESCRIPTOR_VERSION."""
    if not descriptor_path_file.is_file():
        return True
    try:
        descriptor = MobDescriptor.load(descriptor_path_file)
    except Exception:
        return True
    return int(descriptor.version) < DESCRIPTOR_VERSION


def _build_descriptor(asset_name: str, spr_stem: str, _logger) -> None:
    """Build the normal descriptor if missing/stale (skip if up-to-date)."""
    descriptor_path_file = descriptor_path(spr_stem)
    if not _descriptor_needs_rebuild(descriptor_path_file):
        return
    from pybot.recognition.detector.descriptors.descriptor_builder import DescriptorBuilder

    if descriptor_path_file.is_file():
        _logger(
            f"[AUTO-BUILD] {asset_name}: rebuilding stale/invalid descriptor "
            f"({spr_stem}, need version {DESCRIPTOR_VERSION})..."
        )
    else:
        _logger(f"[AUTO-BUILD] {asset_name}: SPR/ACT found, building descriptor ({spr_stem})...")
    builder = DescriptorBuilder(PROJECT_ROOT)
    builder.build(spr_stem, force=True)
    if _descriptor_needs_rebuild(descriptor_path_file):
        raise RuntimeError(
            f"descriptor still missing or below version {DESCRIPTOR_VERSION} after build"
        )
    _logger(f"[AUTO-BUILD] {asset_name}: descriptor ready (v{DESCRIPTOR_VERSION})")


def _build_modified_descriptor(
    asset_name: str, spr_stem: str, builder, _logger
) -> None:
    """Build the big+red modified-sprite descriptor (best-effort)."""
    modified_path = modified_sprite_descriptor_path(spr_stem)
    if not _descriptor_needs_rebuild(modified_path):
        return
    try:
        if modified_path.is_file():
            _logger(
                f"[AUTO-BUILD] {asset_name}: rebuilding stale modified-sprite "
                f"descriptor ({spr_stem})..."
            )
        else:
            _logger(
                f"[AUTO-BUILD] {asset_name}: building modified-sprite "
                f"descriptor ({spr_stem})..."
            )
        builder.build_modified_sprite(spr_stem, force=True)
        if _descriptor_needs_rebuild(modified_path):
            _logger(
                f"[AUTO-BUILD] {asset_name}: modified-sprite descriptor "
                "still missing after build"
            )
        else:
            _logger(
                f"[AUTO-BUILD] {asset_name}: modified-sprite descriptor ready "
                f"(v{DESCRIPTOR_VERSION})"
            )
    except Exception as exc:
        _logger(
            f"[AUTO-BUILD] {asset_name}: modified-sprite descriptor failed — {exc}"
        )


def ensure_mob_assets(*, log_fn: Callable[[str], None] | None = None) -> None:
    """Build or rebuild descriptors that are missing or below DESCRIPTOR_VERSION."""
    _logger = log_fn or print
    if not MOBS_DIR.is_dir():
        _logger(f"[AUTO-BUILD] mob assets folder missing: {MOBS_DIR}")
        return

    pairs = _scan_asset_pairs()
    if not pairs:
        _logger(f"[AUTO-BUILD] no SPR/ACT pairs found under {MOBS_DIR}")
        return

    _logger(
        f"[AUTO-BUILD] checking {len(pairs)} mob(s) "
        f"(descriptor version {DESCRIPTOR_VERSION})..."
    )
    built = 0
    skipped = 0
    failed = 0
    from pybot.recognition.detector.descriptors.descriptor_builder import DescriptorBuilder

    for asset_name, spr_stem in pairs:
        path = descriptor_path(spr_stem)
        needed = _descriptor_needs_rebuild(path)
        try:
            if needed:
                _build_descriptor(asset_name, spr_stem, _logger)
                built += 1
            else:
                skipped += 1
            # Modified-sprite descriptor is independent — check/rebuilt
            # even when the normal descriptor is up-to-date.
            builder = DescriptorBuilder(PROJECT_ROOT)
            _build_modified_descriptor(asset_name, spr_stem, builder, _logger)
        except Exception as exc:
            failed += 1
            _logger(f"[AUTO-BUILD] {asset_name}: build failed — {exc}")

    _logger(
        f"[AUTO-BUILD] done — built/updated={built} up-to-date={skipped} failed={failed}"
    )

    # Sync modified-sprite files into sprite.grf for GRF-modified servers.
    # NOTE: The RO viewer can only handle tables under ~150B compressed / ~130B
    # from EOF, so we only sync shadow files here.  The per-mob sync happens
    # later from bot_controller.py once the hunt target is known.
    from pybot.mobs.sprite_grf import sync_sprite_grf
    try:
        added = sync_sprite_grf(PROJECT_ROOT, logger=_logger)
        if added > 0:
            _logger(f"[AUTO-BUILD] sprite.grf: {added} file(s) synced")
    except Exception as exc:
        _logger(f"[AUTO-BUILD] sprite.grf sync failed — {exc}")


def load_mob_catalog(*, ensure_assets: bool = False) -> list[MobEntry]:
    if ensure_assets:
        ensure_mob_assets()
    if not MOBS_DIR.is_dir():
        return []

    entries: list[MobEntry] = []
    for asset_name, spr_stem in _scan_asset_pairs():
        descriptor_path_file = descriptor_path(spr_stem)
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
