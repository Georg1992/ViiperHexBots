"""Transform a Ragnarok Online .act file.

Applies three edits (file size is preserved):
  1. Death animation actions (the last 8 actions) -> fully transparent (alpha=0).
  2. Every sprite clip in every frame -> scaled 1.5x (scaleX/scaleY multiplied).
  3. Every non-death sprite clip -> recolored red (RGB=255,0,0), alpha preserved.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

SCALE_FACTOR = 1.5
RED = (255, 0, 0)
DEAD_ACTION_COUNT = 8


class Reader:
    def __init__(self, data):
        self.data = data
        self.off = 0

    def u16(self):
        v = struct.unpack_from("<H", self.data, self.off)[0]
        self.off += 2
        return v

    def i32(self):
        v = struct.unpack_from("<i", self.data, self.off)[0]
        self.off += 4
        return v

    def skip(self, n):
        self.off += n


class Clip:
    __slots__ = ("action", "color_off", "scale_offs")

    def __init__(self, action, color_off, scale_offs):
        self.action = action
        self.color_off = color_off
        self.scale_offs = scale_offs


def parse(data):
    r = Reader(data)
    if data[0:2] != b"AC":
        raise ValueError(f"not an ACT file (magic={data[0:2]!r})")
    r.skip(2)
    version = r.u16()
    if not (0x0200 <= version <= 0x0205):
        raise ValueError(f"unsupported ACT version 0x{version:04X}")
    num_actions = r.u16()
    r.skip(10)

    clips = []
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


def transform_bytes(data):
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


def transform(input_path, output_path=None, *, verbose=False):
    input_path = Path(input_path)
    data = input_path.read_bytes()
    out, stats = transform_bytes(data)

    dest = Path(output_path) if output_path is not None else input_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(out)

    if verbose:
        print(f"version           = 0x{stats['version']:04X}")
        print(f"actions           = {stats['num_actions']} (death = {stats['dead_actions']})")
        print(f"clips scaled x{SCALE_FACTOR} = {stats['scaled']}")
        print(f"clips transparent = {stats['transparent']}")
        print(f"clips set red     = {stats['red']}")
        print(f"written           = {dest}")
    return stats


def main():
    ap = argparse.ArgumentParser(description="Transform an RO .act file.")
    ap.add_argument("act", help="path to the .act file")
    ap.add_argument("-o", "--output", help="write transformed ACT here (default: in place)")
    args = ap.parse_args()
    transform(args.act, args.output, verbose=True)


if __name__ == "__main__":
    main()
