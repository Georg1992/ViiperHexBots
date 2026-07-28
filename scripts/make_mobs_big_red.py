#!/usr/bin/env python3
"""Make RO mobs big and red by transforming .act files (SPR is copied as-is).

Edits applied to each .act (file size preserved):
  1. Death animation actions (last 8) -> fully transparent (alpha=0)
  2. Every sprite clip -> scaled 1.5x
  3. Every non-death clip -> recolored red (RGB=255,0,0), alpha preserved

Examples:
  # Single ACT (write beside input or to -o)
  python scripts/make_mobs_big_red.py assets/mobs/Horn/horn.act -o out/horn.act

  # One mob folder (copies .spr, writes transformed .act)
  python scripts/make_mobs_big_red.py assets/mobs/Horn -o out/Horn

  # All mobs under assets/mobs
  python scripts/make_mobs_big_red.py assets/mobs -o out/big_red_mobs --all
"""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path

SCALE_FACTOR = 1.5
RED = (255, 0, 0)
DEAD_ACTION_COUNT = 8


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
    __slots__ = ("action", "color_off", "scale_offs")

    def __init__(self, action: int, color_off: int, scale_offs: list[int]) -> None:
        self.action = action
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
    for a in range(num_actions):
        num_frames = r.i32()
        for _ in range(num_frames):
            r.skip(32)
            num_clips = r.i32()
            for _ in range(num_clips):
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
                clips.append(Clip(a, color_off, scale_offs))
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


def transform_act_bytes(data: bytes) -> tuple[bytes, dict]:
    version, num_actions, clips, end = parse(data)
    if end != len(data):
        raise RuntimeError(
            f"parse mismatch: consumed {end} of {len(data)} bytes; refusing to transform"
        )

    dead_actions = set(range(num_actions - DEAD_ACTION_COUNT, num_actions))
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
    }


def transform_act(input_path: Path, output_path: Path, *, verbose: bool = False) -> dict:
    data = input_path.read_bytes()
    out, stats = transform_act_bytes(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(out)
    if verbose:
        print(f"  ACT  {input_path.name} -> {output_path}")
        print(f"       version=0x{stats['version']:04X} actions={stats['num_actions']}")
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


def process_mob_folder(
    src_dir: Path,
    dst_dir: Path,
    *,
    verbose: bool = False,
) -> int:
    pairs = _find_spr_act_pairs(src_dir)
    if not pairs:
        raise FileNotFoundError(f"no .spr/.act pairs in {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    for spr_path, act_path in pairs:
        out_spr = dst_dir / spr_path.name
        out_act = dst_dir / act_path.name
        shutil.copyfile(spr_path, out_spr)
        if verbose:
            print(f"  SPR  {spr_path.name} -> {out_spr}")
        transform_act(act_path, out_act, verbose=verbose)
    return len(pairs)


def process_mobs_root(
    src_root: Path,
    dst_root: Path,
    *,
    verbose: bool = False,
) -> int:
    mob_dirs = sorted(p for p in src_root.iterdir() if p.is_dir() and _find_spr_act_pairs(p))
    if not mob_dirs:
        raise FileNotFoundError(f"no mob folders with .spr/.act under {src_root}")

    total = 0
    for mob_dir in mob_dirs:
        if verbose:
            print(f"[{mob_dir.name}]")
        total += process_mob_folder(mob_dir, dst_root / mob_dir.name, verbose=verbose)
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Copy SPR and transform ACT to make RO mobs big and red.",
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
        print(f"done — transformed {count} spr/act pair(s)")
        return 0

    if src.is_file() and src.suffix.lower() == ".act":
        dest = args.output.resolve() if args.output is not None else src
        transform_act(src, dest, verbose=verbose)
        print(f"done — wrote {dest}")
        return 0

    if src.is_dir():
        if args.output is None:
            ap.error("--output is required when input is a mob folder")
        count = process_mob_folder(src, args.output.resolve(), verbose=verbose)
        print(f"done — transformed {count} spr/act pair(s) into {args.output}")
        return 0

    ap.error(f"expected .act file or directory, got: {src}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
