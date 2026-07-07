"""Transform a Ragnarok Online .act file.

Applies three edits in place (file size is preserved):
  1. Death animation actions (the last 8 actions) -> fully transparent (alpha=0).
  2. Every sprite clip in every frame -> scaled 1.5x (scaleX/scaleY multiplied).
  3. Every non-death sprite clip -> recolored red (RGB=255,0,0), alpha preserved.

Usage:
    python act_transform.py <file.act>

The script is strict: if the parser does not consume exactly the whole file it
refuses to write anything. Supported ACT versions: 0x0200 - 0x0205.
"""

import argparse
import struct
import sys

SCALE_FACTOR = 1.5
RED = (255, 0, 0)          # R, G, B applied to non-death clips
DEAD_ACTION_COUNT = 8      # the trailing 8 actions are the death animation


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
        self.color_off = color_off      # offset of RGBA color (4 bytes) or None
        self.scale_offs = scale_offs    # list of float offsets to scale


def parse(data):
    """Walk the whole ACT file, collecting clip records. Returns
    (version, num_actions, clips, end_offset)."""
    r = Reader(data)
    if data[0:2] != b"AC":
        raise ValueError(f"not an ACT file (magic={data[0:2]!r})")
    r.skip(2)
    version = r.u16()
    if not (0x0200 <= version <= 0x0205):
        raise ValueError(f"unsupported ACT version 0x{version:04X}")
    num_actions = r.u16()
    r.skip(10)  # reserved

    clips = []
    for a in range(num_actions):
        num_frames = r.i32()
        for _ in range(num_frames):
            r.skip(32)  # range1[4] + range2[4] reserved
            num_clips = r.i32()
            for _ in range(num_clips):
                r.skip(16)  # x, y, sprite number, mirror
                color_off = r.off
                r.skip(4)   # RGBA color
                scale_offs = [r.off]
                r.skip(4)   # scaleX
                if version >= 0x0204:
                    scale_offs.append(r.off)
                    r.skip(4)  # scaleY
                r.skip(8)      # rotation angle + sprite type
                if version >= 0x0205:
                    r.skip(8)  # width + height
                clips.append(Clip(a, color_off, scale_offs))
            r.i32()            # event id (>= 0x0200)
            if version >= 0x0203:
                num_anchor = r.i32()
                r.skip(16 * num_anchor)

    if version >= 0x0201:
        num_sounds = r.i32()
        r.skip(40 * num_sounds)
    if version >= 0x0202:
        r.skip(4 * num_actions)  # per-action frame interval

    return version, num_actions, clips, r.off


def transform(path):
    with open(path, "rb") as fh:
        data = fh.read()

    version, num_actions, clips, end = parse(data)
    if end != len(data):
        raise RuntimeError(
            f"parse mismatch: consumed {end} of {len(data)} bytes; refusing to write")

    dead_actions = set(range(num_actions - DEAD_ACTION_COUNT, num_actions))
    buf = bytearray(data)

    n_transparent = n_scaled = n_red = 0
    for clip in clips:
        # (2) scale every clip
        for off in clip.scale_offs:
            cur = struct.unpack_from("<f", buf, off)[0]
            struct.pack_into("<f", buf, off, cur * SCALE_FACTOR)
        n_scaled += 1

        if clip.action in dead_actions:
            # (1) death animation -> transparent
            buf[clip.color_off + 3] = 0
            n_transparent += 1
        else:
            # (3) all other clips -> red, keep existing alpha
            buf[clip.color_off + 0] = RED[0]
            buf[clip.color_off + 1] = RED[1]
            buf[clip.color_off + 2] = RED[2]
            n_red += 1

    # verify the edited buffer still parses to the exact same size
    _, _, _, end2 = parse(bytes(buf))
    if end2 != len(buf):
        raise RuntimeError("post-edit parse mismatch; not writing")

    with open(path, "wb") as fh:
        fh.write(buf)

    print(f"version           = 0x{version:04X}")
    print(f"actions           = {num_actions} (death = {sorted(dead_actions)})")
    print(f"clips scaled x{SCALE_FACTOR} = {n_scaled}")
    print(f"clips transparent = {n_transparent}")
    print(f"clips set red     = {n_red}")
    print("done.")


def main():
    ap = argparse.ArgumentParser(description="Transform an RO .act file.")
    ap.add_argument("act", help="path to the .act file")
    args = ap.parse_args()
    transform(args.act)


if __name__ == "__main__":
    main()
