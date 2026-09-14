"""Mob descriptor catalog from assets/generated_descriptors/."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pybot.paths import DESCRIPTORS_DIR, MOBS_DIR, PROJECT_ROOT
from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.descriptors.descriptor_builder import (
    DESCRIPTOR_VERSION,
    is_marker_square_descriptor,
)


# These assets ship with the bot and cannot be removed through the UI.
# Each pair shares maps, so they ship as one hunt option. Similar sprites
# (Isilla/Vanberk) collapse to one track; different bodies (Merman/Strouf)
# stay two tracks. GRF marker squares use one color per pair. Member stems
# stay protected so neither can be deleted.
ISILLA_VANBERK_MEMBERS = ("isilla", "vanberk")
ISILLA_VANBERK_OPTION = "isilla+vanberk"
ISILLA_VANBERK_DISPLAY = "Isilla + Vanberk"
MERMAN_STROUF_MEMBERS = ("merman", "strouf")
MERMAN_STROUF_OPTION = "merman+strouf"
MERMAN_STROUF_DISPLAY = "Merman + Strouf"
_PAIRED_HUNTS = (
    (ISILLA_VANBERK_MEMBERS, ISILLA_VANBERK_OPTION, ISILLA_VANBERK_DISPLAY),
    (MERMAN_STROUF_MEMBERS, MERMAN_STROUF_OPTION, MERMAN_STROUF_DISPLAY),
)
_PAIRED_OPTION_MEMBERS = {
    option: members for members, option, _display in _PAIRED_HUNTS
}
_PAIRED_MEMBER_OPTION = {
    member: option
    for members, option, _display in _PAIRED_HUNTS
    for member in members
}
BUILTIN_MOB_ORDER = (
    "wild_rose",
    "desert_wolf",
    "horn",
    "noxious",
    ISILLA_VANBERK_OPTION,
    MERMAN_STROUF_OPTION,
)
BUILTIN_MOB_NAMES = frozenset(
    (
        *BUILTIN_MOB_ORDER,
        *ISILLA_VANBERK_MEMBERS,
        *MERMAN_STROUF_MEMBERS,
    )
)


def is_builtin_mob(name: str) -> bool:
    """Return whether *name* identifies a protected bundled mob."""
    return name.strip().lower() in BUILTIN_MOB_NAMES


def hunt_mob_names(option_name: str) -> tuple[str, ...]:
    """Return the sprite descriptor stems hunted by a catalog option."""
    key = option_name.strip().lower()
    if not key:
        raise ValueError("mob name cannot be empty")
    members = _PAIRED_OPTION_MEMBERS.get(key)
    if members is not None:
        return members
    return (key,)


def hunt_color_key(mob_name: str) -> str:
    """Return the GRF marker-color key for a sprite stem or hunt option.

    Paired map members share one key (``isilla+vanberk``, ``merman+strouf``)
    so their replacement squares render the same color.
    """
    key = mob_name.strip().lower()
    if not key:
        raise ValueError("mob name cannot be empty")
    return _PAIRED_MEMBER_OPTION.get(key, key)


@dataclass(frozen=True)
class MobEntry:
    asset_name: str
    display_name: str
    descriptor_name: str

    @property
    def is_builtin(self) -> bool:
        """True when this entry is one of the mobs shipped with the bot."""
        return is_builtin_mob(self.descriptor_name)


def _collapse_pair(
    entries: list[MobEntry],
    members: tuple[str, ...],
    option: str,
    display: str,
) -> list[MobEntry]:
    present = {entry.descriptor_name.lower() for entry in entries}
    if not set(members).issubset(present):
        return entries
    member_set = set(members)
    collapsed: list[MobEntry] = []
    emitted = False
    for entry in entries:
        if entry.descriptor_name.lower() not in member_set:
            collapsed.append(entry)
            continue
        if emitted:
            continue
        collapsed.append(
            MobEntry(
                asset_name=option,
                display_name=display,
                descriptor_name=option,
            )
        )
        emitted = True
    return collapsed


def collapse_isilla_vanberk(entries: list[MobEntry]) -> list[MobEntry]:
    """Replace separate Isilla and Vanberk radios with one combined option."""
    return _collapse_pair(
        entries,
        ISILLA_VANBERK_MEMBERS,
        ISILLA_VANBERK_OPTION,
        ISILLA_VANBERK_DISPLAY,
    )


def collapse_merman_strouf(entries: list[MobEntry]) -> list[MobEntry]:
    """Replace separate Merman and Strouf radios with one combined option."""
    return _collapse_pair(
        entries,
        MERMAN_STROUF_MEMBERS,
        MERMAN_STROUF_OPTION,
        MERMAN_STROUF_DISPLAY,
    )


def collapse_paired_hunts(entries: list[MobEntry]) -> list[MobEntry]:
    """Replace each complete similar-sprite pair with one combined option."""
    collapsed = entries
    for members, option, display in _PAIRED_HUNTS:
        collapsed = _collapse_pair(collapsed, members, option, display)
    return collapsed


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
    builtin_rank = {name: index for index, name in enumerate(BUILTIN_MOB_ORDER)}
    pairs.sort(
        key=lambda pair: (
            0,
            builtin_rank[_builtin_rank_key(pair[1])],
        )
        if _builtin_rank_key(pair[1]) in builtin_rank
        else (1, pair[0].casefold(), pair[1].casefold())
    )
    return pairs


def _builtin_rank_key(spr_stem: str) -> str:
    key = spr_stem.lower()
    return _PAIRED_MEMBER_OPTION.get(key, key)


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


def _modified_descriptor_needs_rebuild(descriptor_path_file: Path) -> bool:
    """True when the modified descriptor is missing, stale, or not a marker square."""
    if not descriptor_path_file.is_file():
        return True
    try:
        descriptor = MobDescriptor.load(descriptor_path_file)
    except Exception:
        return True
    if int(descriptor.version) < DESCRIPTOR_VERSION:
        return True
    return not is_marker_square_descriptor(descriptor)


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


def _modified_assets_present(asset_name: str, spr_stem: str) -> bool:
    """True when the generated ``modified_sprite/`` pair is a current marker square.

    The pair is the staging area for ``sprite.grf``; the descriptor alone is
    not proof the assets exist.  A missing or stale pair must trigger a
    rebuild even when the descriptor JSON is up to date.
    """
    from pybot.mobs.marker_sprites import marker_assets_are_current

    return marker_assets_are_current(
        MOBS_DIR / asset_name / "modified_sprite",
        spr_stem,
    )


def _build_modified_descriptor(
    asset_name: str, spr_stem: str, builder, _logger
) -> None:
    """Build the colored-square modified-sprite descriptor."""
    modified_path = modified_sprite_descriptor_path(spr_stem)
    if not _modified_descriptor_needs_rebuild(modified_path) and _modified_assets_present(
        asset_name, spr_stem
    ):
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
        if _modified_descriptor_needs_rebuild(modified_path):
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
    """Build descriptors and reconcile the production sprite.grf archive."""
    _logger = log_fn or print

    def _sync_grf() -> None:
        # Run even when the mob catalog is empty so sprite.grf is always
        # present and stale entries are removed after a mob deletion.
        from pybot.mobs.sprite_grf import sync_sprite_grf

        try:
            added = sync_sprite_grf(PROJECT_ROOT, logger=_logger)
            if added > 0:
                _logger(f"[AUTO-BUILD] sprite.grf: {added} file(s) synced")
        except Exception as exc:
            message = f"[AUTO-BUILD] sprite.grf sync failed — {exc}"
            _logger(message)
            raise RuntimeError(message) from exc

    if not MOBS_DIR.is_dir():
        _logger(f"[AUTO-BUILD] mob assets folder missing: {MOBS_DIR}")
        _sync_grf()
        return

    pairs = _scan_asset_pairs()
    if not pairs:
        _logger(f"[AUTO-BUILD] no SPR/ACT pairs found under {MOBS_DIR}")
        _sync_grf()
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
    # from EOF, so only modified shadow files are synced into the archive.
    _sync_grf()


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
    return collapse_paired_hunts(entries)


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
