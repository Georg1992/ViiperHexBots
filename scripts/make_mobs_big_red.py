#!/usr/bin/env python3
"""Generate static, enlarged red RO mob sprites.

For a mob folder, the modified output is deliberately static:

* the first living ACT frame is composited into one canonical SPR frame;
* the output SPR contains exactly that one frame;
* every ACT layer points at SPR frame 0 with the same origin/transform;
* the Die actions (32-39) remain transparent;
* the original ACT action/frame table is preserved for client compatibility.

Keeping the action table matters because the descriptor builder and RO clients
expect the normal action/facing layout.  A one-action ACT would be smaller, but
would break those consumers.  The single-ACT command remains a legacy in-place
ACT transform unless a sibling SPR is available.

Examples:
  # One mob folder (writes one-frame SPR + static ACT)
  python scripts/make_mobs_big_red.py assets/mobs/Horn -o out/Horn

  # All mobs under assets/mobs
  python scripts/make_mobs_big_red.py assets/mobs -o out/big_red_mobs --all

  # Legacy ACT-only transform
  python scripts/make_mobs_big_red.py assets/mobs/Horn/horn.act -o out/horn.act
"""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from pybot.recognition.act_reader import ActFile, ActFrameRef, ActReader
from pybot.recognition.frame_renderer import render_act_frame
from pybot.recognition.spr_reader import SprFile, SprReader

SCALE_FACTOR = 1.5
RED = (255, 0, 0)
# Ragnarok monster sprites use a fixed action layout: 0-7 stand, 8-15 walk,
# 16-23 attack, 24-31 hit, 32-39 die (death). Some mobs add living special
# actions after 39 (e.g. idle poses 40-47), so the death range is fixed here
# instead of derived from the total action count.
DEAD_ACTION_FIRST = 32
DEAD_ACTION_LAST_EXCLUSIVE = 40


class Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.off = 0

    def u16(self) -> int:
        v = struct.unpack_from("<H", self.data, self.off)[0]
        self.off += 2
        return v

    def i32(self) -> int:
        v = struct.unpack_from("<i", self.data, self.off)[0]
        self.off += 4
        return v

    def skip(self, n: int) -> None:
        self.off += n


class Clip:
    __slots__ = ("action", "layer", "layer_off", "color_off", "scale_offs")

    def __init__(
        self,
        action: int,
        layer: int,
        layer_off: int,
        color_off: int,
        scale_offs: list[int],
    ) -> None:
        self.action = action
        self.layer = layer
        self.layer_off = layer_off
        self.color_off = color_off
        self.scale_offs = scale_offs


def parse(data: bytes) -> tuple[int, int, list[Clip], int]:
    r = Reader(data)
    if data[0:2] != b"AC":
        raise ValueError(f"not an ACT file (magic={data[0:2]!r})")
    r.skip(2)
    version = r.u16()
    if not (0x0200 <= version <= 0x0205):
        raise ValueError(f"unsupported ACT version 0x{version:04X}")
    num_actions = r.u16()
    r.skip(10)

    clips: list[Clip] = []
    for action in range(num_actions):
        num_frames = r.i32()
        for _ in range(num_frames):
            r.skip(32)
            num_clips = r.i32()
            for layer in range(num_clips):
                layer_off = r.off
                r.skip(16)
                color_off = r.off
                r.skip(4)
                scale_offs = [r.off]
                r.skip(4)
                if version >= 0x0204:
                    scale_offs.append(r.off)
                    r.skip(4)
                r.skip(8)
                if version >= 0x0205:
                    r.skip(8)
                clips.append(Clip(action, layer, layer_off, color_off, scale_offs))
            r.i32()
            if version >= 0x0203:
                num_anchor = r.i32()
                r.skip(16 * num_anchor)

    if version >= 0x0201:
        num_sounds = r.i32()
        r.skip(40 * num_sounds)
    if version >= 0x0202:
        r.skip(4 * num_actions)

    return version, num_actions, clips, r.off


def _dead_actions(num_actions: int) -> set[int]:
    """Return the fixed RO death action range, clamped to what exists.

    Death is always actions 32-39 regardless of how many actions the sprite
    has: mobs with extra living actions after 39 (e.g. desert wolf's specials
    at 40-47) must not be mistaken for death, and truncated sprites only
    expose the death actions they actually contain.
    """
    return set(
        range(DEAD_ACTION_FIRST, min(DEAD_ACTION_LAST_EXCLUSIVE, num_actions))
    )


def transform_act_bytes(data: bytes) -> tuple[bytes, dict]:
    """Apply the original big/red ACT-only transform without changing layout."""
    version, num_actions, clips, end = parse(data)
    if end != len(data):
        raise RuntimeError(
            f"parse mismatch: consumed {end} of {len(data)} bytes; refusing to transform"
        )

    dead_actions = _dead_actions(num_actions)
    buf = bytearray(data)
    n_transparent = n_scaled = n_red = 0
    for clip in clips:
        for off in clip.scale_offs:
            cur = struct.unpack_from("<f", buf, off)[0]
            struct.pack_into("<f", buf, off, cur * SCALE_FACTOR)
        n_scaled += 1

        if clip.action in dead_actions:
            buf[clip.color_off + 3] = 0
            n_transparent += 1
        else:
            buf[clip.color_off + 0] = RED[0]
            buf[clip.color_off + 1] = RED[1]
            buf[clip.color_off + 2] = RED[2]
            n_red += 1

    _, _, _, end2 = parse(bytes(buf))
    if end2 != len(buf):
        raise RuntimeError("post-edit parse mismatch; not writing")

    return bytes(buf), {
        "version": version,
        "num_actions": num_actions,
        "dead_actions": sorted(dead_actions),
        "scaled": n_scaled,
        "transparent": n_transparent,
        "red": n_red,
        "static": False,
    }


def _canonical_frame(act_file: ActFile) -> ActFrameRef:
    """Return the first renderable living frame used for the static sprite."""
    dead_actions = _dead_actions(len(act_file.actions))
    for action in act_file.actions:
        if action.index in dead_actions:
            continue
        for frame in action.frames:
            if frame.layers:
                return frame
    raise RuntimeError("ACT has no renderable living frame")


def _canonical_origin(frame: ActFrameRef) -> tuple[int, int]:
    return (
        min(layer.x for layer in frame.layers),
        min(layer.y for layer in frame.layers),
    )


def _static_source_frame(frame: ActFrameRef) -> ActFrameRef:
    """Apply the existing big/red ACT transform to the canonical frame."""
    return replace(
        frame,
        layers=[
            replace(
                layer,
                color_tint=(RED[0], RED[1], RED[2], 255),
                scale_x=layer.scale_x * SCALE_FACTOR,
                scale_y=layer.scale_y * SCALE_FACTOR,
            )
            for layer in frame.layers
        ],
    )


def _palette_bytes(spr_file: SprFile) -> bytes:
    raw = bytearray()
    for color in spr_file.palette:
        raw.extend((color.red, color.green, color.blue, color.alpha))
    return bytes(raw)


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


def _tinted_palette(spr_file: SprFile) -> list[tuple[int, int, int, int]]:
    """Return the palette with the red tint applied (BGR + alpha).

    Mirrors the ACT layer tint semantics used by the live client: RGB is
    multiplied by the tint colour (red) per channel, so blue/green drop to 0
    and red is preserved.  Index 0 stays the transparent slot with alpha 0;
    every other entry becomes opaque so the indexed frame renders solid.
    """
    tinted: list[tuple[int, int, int, int]] = []
    for index, color in enumerate(spr_file.palette):
        if index == 0:
            tinted.append((0, 0, 0, 0))
            continue
        red = int(color.red)
        tinted.append((0, 0, red, 255))
    return tinted


def _encode_static_spr(spr_file: SprFile, bgra: np.ndarray) -> bytes:
    """Encode one canonical frame as an indexed SPR 2.1 with a red palette.

    The RO client renders indexed frames upright with the file palette; RGBA
    frames are unreliable in the stock client (they can appear colour-swapped
    and flipped).  The red tint is therefore baked into the palette and every
    opaque pixel is mapped to its nearest palette shade.
    """
    height, width = bgra.shape[:2]
    if not (1 <= width <= 32767 and 1 <= height <= 32767):
        raise RuntimeError(f"canonical sprite is too large: {width}x{height}")

    tinted = _tinted_palette(spr_file)
    candidates = np.asarray(
        [list(color[:3]) for color in tinted[1:]], dtype=np.int16
    )

    opaque = bgra[:, :, 3] >= 128
    opaque_flat = opaque.reshape(-1)
    indices = np.zeros(width * height, dtype=np.uint8)
    # Map each opaque pixel to the nearest red-tinted palette shade.
    if opaque.any():
        unique, inverse = np.unique(
            bgra[:, :, :3][opaque], axis=0, return_inverse=True
        )
        opaque_positions = np.flatnonzero(opaque_flat)
        for local_index, color in enumerate(unique.astype(np.int16)):
            distance = ((candidates - color) ** 2).sum(axis=1)
            best = int(np.argmin(distance))
            indices[opaque_positions[inverse == local_index]] = best + 1
    indices = bytes(indices)

    encoded = _encode_rle(indices)
    if len(encoded) > 65535:
        raise RuntimeError(f"RLE sprite frame too large for SPR 2.1: {len(encoded)} bytes")

    palette_out = bytearray()
    for entry in tinted:
        palette_out.extend((entry[2], entry[1], entry[0], entry[3]))
    return (
        struct.pack("<2sHHHhhH", b"SP", 0x0201, 1, 0, width, height, len(encoded))
        + encoded
        + bytes(palette_out)
    )


STATIC_ACT_VERSION = 0x0200


def make_static_act_bytes(
    act_file: ActFile,
    *,
    origin: tuple[int, int],
    width: int,
    height: int,
) -> tuple[bytes, dict]:
    """Build a compact static ACT with the original action/frame counts.

    Rebuilding the frame records, instead of patching them in place, also
    handles source actions that contain zero layers (notably Noxious). Every
    output frame receives exactly one layer, so the game can never select a
    missing or animated source frame. ACT 2.0 is sufficient for the static
    records and avoids carrying source-only events/attachments forward.
    """
    num_actions = len(act_file.actions)
    dead_actions = _dead_actions(num_actions)
    x, y = origin
    buf = bytearray(struct.pack("<2sHH10x", b"AC", STATIC_ACT_VERSION, num_actions))

    for action in act_file.actions:
        buf.extend(struct.pack("<I", len(action.frames)))
        for _frame in action.frames:
            # attack range + fit range (unused by recognition/client rendering)
            buf.extend(b"\0" * 32)
            buf.extend(struct.pack("<I", 1))  # exactly one composite layer
            buf.extend(struct.pack("<iiii", x, y, 0, 0))
            alpha = 0 if action.index in dead_actions else 255
            buf.extend(struct.pack("<BBBB", 255, 255, 255, alpha))
            buf.extend(struct.pack("<fii", 1.0, 0, 0))
            buf.extend(struct.pack("<I", 0))  # event id

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


def transform_act(
    input_path: Path,
    output_path: Path,
    *,
    verbose: bool = False,
) -> dict:
    """Apply the legacy in-place ACT-only big/red transform."""
    data = input_path.read_bytes()
    out, stats = transform_act_bytes(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(out)
    if verbose:
        print(f"  ACT  {input_path.name} -> {output_path}")
        print(f"       version=0x{stats['version']:04X} actions={stats['num_actions']}")
        if stats.get("static"):
            print(
                f"       static frame=0 size={stats['canonical_width']}x"
                f"{stats['canonical_height']} transparent-death-actions=yes"
            )
        else:
            print(
                f"       scaled={stats['scaled']} red={stats['red']} "
                f"transparent={stats['transparent']}"
            )
    return stats


def _find_spr_act_pairs(folder: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for spr_path in sorted(folder.glob("*.spr")):
        act_path = folder / f"{spr_path.stem}.act"
        if act_path.is_file():
            pairs.append((spr_path, act_path))
    return pairs


def generate_static_pair(
    spr_path: Path,
    act_path: Path,
    out_spr: Path,
    out_act: Path,
    *,
    verbose: bool = False,
) -> dict:
    """Generate one static SPR/ACT pair and return its ACT statistics."""
    spr_file = SprReader(spr_path).load()
    act_file = ActReader(act_path).load()
    canonical = _canonical_frame(act_file)
    composite = render_act_frame(spr_file, _static_source_frame(canonical))

    out_spr.parent.mkdir(parents=True, exist_ok=True)
    out_spr.write_bytes(_encode_static_spr(spr_file, composite))
    static_act, stats = make_static_act_bytes(
        act_file,
        origin=_canonical_origin(canonical),
        width=int(composite.shape[1]),
        height=int(composite.shape[0]),
    )
    out_act.parent.mkdir(parents=True, exist_ok=True)
    out_act.write_bytes(static_act)

    if verbose:
        print(
            f"  SPR  {spr_path.name} -> {out_spr} "
            f"(one frame, {composite.shape[1]}x{composite.shape[0]})"
        )
        print(f"  ACT  {act_path.name} -> {out_act}")
        print(
            f"       version=0x{stats['version']:04X} actions={stats['num_actions']} "
            f"static frame=0 transparent-death-actions=yes"
        )
    return stats


def process_mob_folder(
    src_dir: Path,
    dst_dir: Path,
    *,
    verbose: bool = False,
) -> int:
    """Generate static one-frame modified assets for every SPR/ACT pair."""
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate static one-frame big/red SPR+ACT mob assets.",
    )
    ap.add_argument(
        "input",
        type=Path,
        help=".act file, one mob folder, or (with --all) a root of mob folders",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output .act path, mob folder, or root folder (required for folders / --all)",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="treat input as a root of mob folders (e.g. assets/mobs)",
    )
    ap.add_argument("-q", "--quiet", action="store_true", help="less logging")
    args = ap.parse_args(argv)

    src = args.input.resolve()
    verbose = not args.quiet

    if args.all:
        if args.output is None:
            ap.error("--output is required with --all")
        if not src.is_dir():
            ap.error(f"--all expects a directory: {src}")
        count = process_mobs_root(src, args.output.resolve(), verbose=verbose)
        print(f"done — generated {count} static spr/act pair(s)")
        return 0

    if src.is_file() and src.suffix.lower() == ".act":
        dest = args.output.resolve() if args.output is not None else src
        sibling_spr = src.with_suffix(".spr")
        if sibling_spr.is_file():
            generate_static_pair(
                sibling_spr,
                src,
                dest.with_suffix(".spr"),
                dest,
                verbose=verbose,
            )
            print(f"done — wrote static pair {dest.with_suffix('.spr')} and {dest}")
        else:
            transform_act(src, dest, verbose=verbose)
            print(f"done — wrote {dest}")
        return 0

    if src.is_dir():
        if args.output is None:
            ap.error("--output is required when input is a mob folder")
        count = process_mob_folder(src, args.output.resolve(), verbose=verbose)
        print(f"done — generated {count} static spr/act pair(s) into {args.output}")
        return 0

    ap.error(f"expected .act file or directory, got: {src}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
