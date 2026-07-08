"""Mob descriptor catalog from assets/generated_descriptors/."""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path

from pybot.paths import PROJECT_ROOT

DESCRIPTOR_ROOT = PROJECT_ROOT / "assets" / "generated_descriptors"
ASSET_ROOT = PROJECT_ROOT / "assets" / "mobs"
MODIFIED_ROOT = PROJECT_ROOT / "assets" / "modified_mobs"
ACT_TRANSFORM_PATH = PROJECT_ROOT / "assets" / "act_transform.py"


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


def _load_act_transform():
    """Load assets/act_transform.py as a module (assets/ is not a package)."""
    spec = importlib.util.spec_from_file_location("act_transform", ACT_TRANSFORM_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load act_transform from {ACT_TRANSFORM_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_descriptor(asset_name: str, spr_stem: str, _logger) -> None:
    """Compile a descriptor for a mob that doesn't have one yet."""
    descriptor_path = DESCRIPTOR_ROOT / spr_stem / "simple" / "descriptor.json"
    if descriptor_path.is_file():
        return
    # Lazy import — heavy deps only needed for a real build. Importing
    # _mob_rec_path puts mob-recognition/ and mob-recognition/simple/ on
    # sys.path so ``descriptors`` resolves.
    import pybot.runtime._mob_rec_path  # noqa: F401 — sets up mob-recognition sys.path
    from descriptors.descriptor_builder import SimpleDescriptorBuilder  # type: ignore[import-untyped]

    _logger(f"[AUTO-BUILD] {asset_name}: SPR/ACT found, building descriptor ({spr_stem})...")
    SimpleDescriptorBuilder(PROJECT_ROOT).build(spr_stem, force=True)
    _logger(f"[AUTO-BUILD] {asset_name}: descriptor ready")


def _build_modified_mob(asset_name: str, spr_stem: str, _logger) -> None:
    """Mirror one mob folder into modified_mobs/ with transformed SPR/ACT.

    The SPR is copied verbatim and the ACT is run through ``act_transform``.
    Original assets are never touched.
    """
    target_dir = MODIFIED_ROOT / asset_name
    target_act = target_dir / f"{spr_stem}.act"
    target_spr = target_dir / f"{spr_stem}.spr"
    if target_act.is_file() and target_spr.is_file():
        return

    src_dir = ASSET_ROOT / asset_name
    target_dir.mkdir(parents=True, exist_ok=True)
    _logger(f"[MODIFY] {asset_name}: creating modified SPR/ACT ({spr_stem})...")
    shutil.copyfile(src_dir / f"{spr_stem}.spr", target_spr)
    _load_act_transform().transform(src_dir / f"{spr_stem}.act", target_act)
    _logger(f"[MODIFY] {asset_name}: modified assets ready")


def ensure_mob_assets(*, log_fn: "Callable[[str], None] | None" = None) -> None:
    """Bring generated assets in sync with the mob source folders.

    For each mob folder under assets/mobs/ that has an SPR/ACT pair, ensure a
    compiled descriptor exists and a transformed mirror exists under
    assets/modified_mobs/. Both steps are keyed on missing output, so this
    mirrors any newly added mob folder and is a no-op once everything is built.
    """
    _logger = log_fn or print
    if not ASSET_ROOT.is_dir():
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


def load_mob_catalog() -> list[MobEntry]:
    ensure_mob_assets()
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
