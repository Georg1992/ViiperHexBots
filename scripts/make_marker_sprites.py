#!/usr/bin/env python3
"""Generate static colored-square RO mob sprites for sprite.grf.

Examples:
  python scripts/make_marker_sprites.py assets/mobs/Horn/sprite -o out/Horn
  python scripts/make_marker_sprites.py assets/mobs -o out/marker_mobs --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pybot.mobs.marker_sprites import (
    generate_static_pair,
    process_mob_folder,
    process_mobs_root,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate static colored-square SPR+ACT mob assets.",
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
        sibling_spr = src.with_suffix(".spr")
        if not sibling_spr.is_file():
            ap.error(f"marker squares require a sibling SPR: {sibling_spr}")
        if args.output is None:
            ap.error("--output is required when input is an .act file")
        dest = args.output.resolve()
        generate_static_pair(
            sibling_spr,
            src,
            dest.with_suffix(".spr"),
            dest,
            verbose=verbose,
        )
        print(f"done — wrote static pair {dest.with_suffix('.spr')} and {dest}")
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
