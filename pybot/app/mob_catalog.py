"""Mob descriptor catalog from generated_descriptors/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pybot.paths import PROJECT_ROOT

DESCRIPTOR_ROOT = PROJECT_ROOT / "generated_descriptors"
ASSET_ROOT = PROJECT_ROOT / "assets" / "mobs"


@dataclass(frozen=True)
class MobEntry:
    asset_name: str       # Folder name in assets/mobs/ (e.g. "DesertWolf")
    display_name: str     # Human-readable label (e.g. "Desert Wolf")
    descriptor_name: str  # Lowercase stem used for descriptor lookup (e.g. "desert_wolf")


def mob_display_name(asset_name: str) -> str:
    display = asset_name.replace("_", " ").replace("-", " ")
    if not display:
        return asset_name
    return display[0].upper() + display[1:]


def _scan_asset_pairs() -> list[tuple[str, str]]:
    """Scan assets/mobs/ for folders containing SPR+ACT file pairs.

    Returns list of (asset_folder_name, spr_stem) tuples, e.g.
    ("DesertWolf", "desert_wolf").
    The spr_stem (lowercase) is used as the descriptor name.
    """
    if not ASSET_ROOT.is_dir():
        return []
    pairs: list[tuple[str, str]] = []
    for mob_dir in sorted(ASSET_ROOT.iterdir()):
        if not mob_dir.is_dir():
            continue
        for spr_path in sorted(mob_dir.glob("*.spr")):
            spr_stem = spr_path.stem
            act_path = mob_dir / f"{spr_stem}.act"
            if act_path.is_file():
                pairs.append((mob_dir.name, spr_stem))
                break  # one pair per folder
    return pairs


def ensure_descriptors(*, log_fn: "Callable[[str], None] | None" = None) -> None:
    """Auto-build descriptor JSON for any mob with SPR/ACT assets but no compiled descriptor."""
    _logger = log_fn or print
    built_any = False
    for asset_name, spr_stem in _scan_asset_pairs():
        descriptor_path = DESCRIPTOR_ROOT / spr_stem / "simple" / "descriptor.json"
        if descriptor_path.is_file():
            continue
        # Lazy imports — heavy dependencies only needed for builds
        try:
            import sys as _sys
            mob_rec_dir = str(Path(__file__).resolve().parents[2] / "mob-recognition")
            if mob_rec_dir not in _sys.path:
                _sys.path.insert(0, mob_rec_dir)
            from descriptors.descriptor_builder import SimpleDescriptorBuilder  # type: ignore[import-untyped]

            _logger(f"[AUTO-BUILD] {asset_name}: SPR/ACT found, building descriptor ({spr_stem})...")
            SimpleDescriptorBuilder(PROJECT_ROOT).build(spr_stem, force=True)
            _logger(f"[AUTO-BUILD] {asset_name}: descriptor ready")
            built_any = True
        except Exception as exc:
            _logger(f"[AUTO-BUILD] {asset_name}: build failed — {exc}")

def load_mob_catalog() -> list[MobEntry]:
    ensure_descriptors()
    if not ASSET_ROOT.is_dir():
        return []

    entries: list[MobEntry] = []
    for asset_name, spr_stem in _scan_asset_pairs():
        descriptor_path = DESCRIPTOR_ROOT / spr_stem / "simple" / "descriptor.json"
        if not descriptor_path.is_file():
            continue
        entries.append(MobEntry(
            asset_name=asset_name,
            display_name=mob_display_name(asset_name),
            descriptor_name=spr_stem,
        ))
    return entries


def mob_folder_by_index(catalog: list[MobEntry], index: int) -> str:
    if not catalog:
        return "horn"
    clamped = max(1, min(index, len(catalog)))
    return catalog[clamped - 1].descriptor_name
