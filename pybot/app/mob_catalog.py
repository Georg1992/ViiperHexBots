"""Mob descriptor catalog from generated_descriptors/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pybot.paths import PROJECT_ROOT

DESCRIPTOR_ROOT = PROJECT_ROOT / "generated_descriptors"
ASSET_ROOT = PROJECT_ROOT / "assets" / "mobs"


@dataclass(frozen=True)
class MobEntry:
    folder_name: str
    display_name: str


def mob_display_name(folder_name: str) -> str:
    display = folder_name.replace("_", " ").replace("-", " ")
    if not display:
        return folder_name
    return display[0].upper() + display[1:]


def ensure_descriptors(*, log_fn: "Callable[[str], None] | None" = None) -> None:
    """Auto-build descriptor JSON for any mob with SPR/ACT assets but no compiled descriptor."""
    _logger = log_fn or print
    if not ASSET_ROOT.is_dir():
        return
    built_any = False
    for mob_dir in sorted(ASSET_ROOT.iterdir()):
        if not mob_dir.is_dir():
            continue
        mob_name = mob_dir.name.lower()
        spr = mob_dir / f"{mob_name}.spr"
        act = mob_dir / f"{mob_name}.act"
        if not spr.is_file() or not act.is_file():
            continue
        descriptor_path = DESCRIPTOR_ROOT / mob_name / "simple" / "descriptor.json"
        if descriptor_path.is_file():
            continue
        # Lazy imports — heavy dependencies only needed for builds
        try:
            import sys as _sys
            mob_rec_dir = str(Path(__file__).resolve().parents[2] / "mob-recognition")
            if mob_rec_dir not in _sys.path:
                _sys.path.insert(0, mob_rec_dir)
            from descriptors.descriptor_builder import SimpleDescriptorBuilder  # type: ignore[import-untyped]

            _logger(f"[AUTO-BUILD] {mob_name}: SPR/ACT found, building descriptor...")
            SimpleDescriptorBuilder(PROJECT_ROOT).build(mob_name, force=True)
            _logger(f"[AUTO-BUILD] {mob_name}: descriptor ready")
            built_any = True
        except Exception as exc:
            _logger(f"[AUTO-BUILD] {mob_name}: build failed — {exc}")


def load_mob_catalog(root: Path | None = None) -> list[MobEntry]:
    ensure_descriptors()
    descriptor_root = root or DESCRIPTOR_ROOT
    if not descriptor_root.is_dir():
        return []

    entries: list[MobEntry] = []
    for mob_dir in sorted(descriptor_root.iterdir()):
        if not mob_dir.is_dir():
            continue
        descriptor_path = mob_dir / "simple" / "descriptor.json"
        if not descriptor_path.is_file():
            continue
        folder_name = mob_dir.name
        entries.append(MobEntry(folder_name=folder_name, display_name=mob_display_name(folder_name)))
    return entries


def mob_folder_by_index(catalog: list[MobEntry], index: int) -> str:
    if not catalog:
        return "horn"
    clamped = max(1, min(index, len(catalog)))
    return catalog[clamped - 1].folder_name
