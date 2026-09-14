"""Generate static colored-square SPR+ACT replacements for sprite.grf.

Each hunted mob is replaced by one opaque square of a distinctive color.
Paired map hunts share a color so both members are the same marker; every
other mob gets its own color. Death actions stay transparent. The original
ACT action/frame table is preserved so the RO client keeps a valid layout.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np

from pybot.mobs.catalog import (
    ISILLA_VANBERK_OPTION,
    MERMAN_STROUF_OPTION,
    hunt_color_key,
)
from pybot.recognition.act_reader import ActFile, ActReader
from pybot.recognition.spr_reader import SprParseError, SprReader

MARKER_SPRITE_SIZE = 64
DEAD_ACTION_FIRST = 32
DEAD_ACTION_LAST_EXCLUSIVE = 40
STATIC_ACT_VERSION = 0x0200

# Reserved RGB colors for shipped hunt keys. Values stay far apart in BGR so
# the GRF heatmap of one hunt does not light up another hunt's square.
_BUILTIN_MARKER_RGB: dict[str, tuple[int, int, int]] = {
    "wild_rose": (255, 0, 0),
    "desert_wolf": (255, 220, 0),
    "horn": (0, 220, 255),
    "noxious": (255, 0, 220),
    ISILLA_VANBERK_OPTION: (255, 110, 0),
    MERMAN_STROUF_OPTION: (160, 0, 255),
}

# Palette for imported mobs. Distinct from the reserved builtin colors.
_CUSTOM_MARKER_RGB: tuple[tuple[int, int, int], ...] = (
    (255, 64, 128),
    (192, 255, 0),
    (64, 255, 192),
    (192, 64, 255),
    (255, 192, 64),
    (64, 128, 255),
    (255, 32, 32),
    (32, 255, 160),
    (255, 160, 255),
    (160, 255, 32),
)


def modified_sprite_rgb(mob_name: str) -> tuple[int, int, int]:
    """Return the RGB marker color for a sprite stem or hunt option."""
    key = hunt_color_key(mob_name)
    reserved = _BUILTIN_MARKER_RGB.get(key)
    if reserved is not None:
        return reserved
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    index = int.from_bytes(digest[:2], "big") % len(_CUSTOM_MARKER_RGB)
    return _CUSTOM_MARKER_RGB[index]


def _dead_actions(num_actions: int) -> set[int]:
    """Return the fixed RO death action range, clamped to what exists."""
    return set(
        range(DEAD_ACTION_FIRST, min(DEAD_ACTION_LAST_EXCLUSIVE, num_actions))
    )


def _encode_rle(indices: bytes) -> bytes:
    """Encode indexed pixels using the SPR 2.1 zero-run encoding."""
    encoded = bytearray()
    i = 0
    while i < len(indices):
        value = indices[i]
        if value == 0:
            end = i + 1
            while end < len(indices) and indices[end] == 0 and end - i < 255:
                end += 1
            encoded.extend((0, end - i))
            i = end
        else:
            encoded.append(value)
            i += 1
    return bytes(encoded)


def encode_marker_spr(rgb: tuple[int, int, int], size: int = MARKER_SPRITE_SIZE) -> bytes:
    """Encode one opaque square as an indexed SPR 2.1 file."""
    if not (1 <= size <= 32767):
        raise RuntimeError(f"marker sprite is too large: {size}x{size}")
    red, green, blue = rgb
    indices = bytes([1]) * (size * size)
    encoded = _encode_rle(indices)
    if len(encoded) > 65535:
        raise RuntimeError(f"RLE sprite frame too large for SPR 2.1: {len(encoded)} bytes")

    palette = bytearray(1024)
    palette[4:8] = bytes((red, green, blue, 255))
    return (
        struct.pack("<2sHHHhhH", b"SP", 0x0201, 1, 0, size, size, len(encoded))
        + encoded
        + bytes(palette)
    )


def make_static_act_bytes(
    act_file: ActFile,
    *,
    origin: tuple[int, int],
    width: int,
    height: int,
) -> tuple[bytes, dict]:
    """Build a compact static ACT with the original action/frame counts."""
    num_actions = len(act_file.actions)
    dead_actions = _dead_actions(num_actions)
    x, y = origin
    buf = bytearray(struct.pack("<2sHH10x", b"AC", STATIC_ACT_VERSION, num_actions))

    for action in act_file.actions:
        buf.extend(struct.pack("<I", len(action.frames)))
        for _frame in action.frames:
            buf.extend(b"\0" * 32)
            buf.extend(struct.pack("<I", 1))
            buf.extend(struct.pack("<iiii", x, y, 0, 0))
            alpha = 0 if action.index in dead_actions else 255
            buf.extend(struct.pack("<BBBB", 255, 255, 255, alpha))
            buf.extend(struct.pack("<fii", 1.0, 0, 0))
            buf.extend(struct.pack("<I", 0))

    return bytes(buf), {
        "version": STATIC_ACT_VERSION,
        "source_version": act_file.version,
        "num_actions": num_actions,
        "dead_actions": sorted(dead_actions),
        "static": True,
        "spr_frame_count": 1,
        "canonical_width": width,
        "canonical_height": height,
    }


def marker_origin(size: int = MARKER_SPRITE_SIZE) -> tuple[int, int]:
    """Place the square so it stands on the action origin (feet)."""
    return (-size // 2, -size)


def generate_static_pair(
    spr_path: Path,
    act_path: Path,
    out_spr: Path,
    out_act: Path,
    *,
    verbose: bool = False,
) -> dict:
    """Write one colored-square SPR/ACT pair and return ACT statistics."""
    act_file = ActReader(act_path).load()
    rgb = modified_sprite_rgb(spr_path.stem)
    size = MARKER_SPRITE_SIZE
    origin = marker_origin(size)

    out_spr.parent.mkdir(parents=True, exist_ok=True)
    out_spr.write_bytes(encode_marker_spr(rgb, size))
    static_act, stats = make_static_act_bytes(
        act_file,
        origin=origin,
        width=size,
        height=size,
    )
    out_act.parent.mkdir(parents=True, exist_ok=True)
    out_act.write_bytes(static_act)
    stats["rgb"] = rgb
    stats["marker_size"] = size

    if verbose:
        print(
            f"  SPR  {spr_path.name} -> {out_spr} "
            f"(one {size}x{size} square rgb={rgb})"
        )
        print(f"  ACT  {act_path.name} -> {out_act}")
        print(
            f"       version=0x{stats['version']:04X} actions={stats['num_actions']} "
            f"static frame=0 transparent-death-actions=yes"
        )
    return stats


def _find_spr_act_pairs(folder: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for spr_path in sorted(folder.glob("*.spr")):
        act_path = folder / f"{spr_path.stem}.act"
        if act_path.is_file():
            pairs.append((spr_path, act_path))
    return pairs


def process_mob_folder(
    src_dir: Path,
    dst_dir: Path,
    *,
    verbose: bool = False,
) -> int:
    """Generate static marker squares for every SPR/ACT pair."""
    pairs = _find_spr_act_pairs(src_dir)
    if not pairs:
        raise FileNotFoundError(f"no .spr/.act pairs in {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    for spr_path, act_path in pairs:
        generate_static_pair(
            spr_path,
            act_path,
            dst_dir / spr_path.name,
            dst_dir / act_path.name,
            verbose=verbose,
        )
    return len(pairs)


def process_mobs_root(
    src_root: Path,
    dst_root: Path,
    *,
    verbose: bool = False,
) -> int:
    """Generate every mob, accepting either direct or ``*/sprite`` assets."""
    entries: list[tuple[Path, str]] = []
    for mob_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        if _find_spr_act_pairs(mob_dir):
            entries.append((mob_dir, mob_dir.name))
            continue
        sprite_dir = mob_dir / "sprite"
        if sprite_dir.is_dir() and _find_spr_act_pairs(sprite_dir):
            entries.append((sprite_dir, mob_dir.name))
    if not entries:
        raise FileNotFoundError(f"no mob folders with .spr/.act under {src_root}")

    total = 0
    for asset_dir, output_name in entries:
        if verbose:
            print(f"[{output_name}]")
        total += process_mob_folder(asset_dir, dst_root / output_name, verbose=verbose)
    return total


def marker_assets_are_current(modified_dir: Path, spr_stem: str) -> bool:
    """True when the on-disk pair is a square of this mob's marker color."""
    spr_path = modified_dir / f"{spr_stem}.spr"
    act_path = modified_dir / f"{spr_stem}.act"
    if not spr_path.is_file() or not act_path.is_file():
        return False
    try:
        spr = SprReader(spr_path).load()
    except SprParseError:
        return False
    if spr.frame_count != 1:
        return False
    frame = spr.get_frame(0)
    if frame is None:
        return False
    if frame.width != MARKER_SPRITE_SIZE or frame.height != MARKER_SPRITE_SIZE:
        return False
    opaque = frame.rgba[:, :, 3] >= 128
    if not bool(np.all(opaque)):
        return False
    red, green, blue = modified_sprite_rgb(spr_stem)
    expected_bgr = np.array([blue, green, red], dtype=np.uint8)
    return bool(np.all(frame.rgba[:, :, :3] == expected_bgr))
