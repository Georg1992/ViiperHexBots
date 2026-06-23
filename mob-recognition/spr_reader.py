"""
Ragnarok Online .spr file reader.

Supports SPR versions 1.0, 1.1, 2.0, and 2.1 (indexed + optional RGBA + palette).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from binary_reader import BinaryReader


class SprParseError(Exception):
    """Raised when a .spr file cannot be parsed."""


@dataclass
class SprFrame:
    """One decoded sprite frame ready for PNG export."""

    index: int
    width: int
    height: int
    rgba: np.ndarray
    offset_x: int = 0
    offset_y: int = 0
    is_rgba: bool = False


@dataclass
class SprPaletteColor:
    blue: int
    green: int
    red: int
    alpha: int


@dataclass
class SprFile:
    path: Path
    version: int
    indexed_frames: List[Optional[SprFrame]]
    rgba_frames: List[SprFrame]
    palette: List[SprPaletteColor]

    @property
    def frame_count(self) -> int:
        return len(self.indexed_frames) + len(self.rgba_frames)

    def get_frame(self, index: int) -> Optional[SprFrame]:
        if 0 <= index < len(self.indexed_frames):
            return self.indexed_frames[index]
        rgba_index = index - len(self.indexed_frames)
        if 0 <= rgba_index < len(self.rgba_frames):
            return self.rgba_frames[rgba_index]
        return None


def decompress_rle_indices(chunk: bytes, pixel_count: int) -> bytes:
    """Decode RO SPR 2.1 RLE into palette indices."""
    out = bytearray(pixel_count)
    pos = 0
    i = 0
    while pos < pixel_count and i < len(chunk):
        value = chunk[i]
        i += 1
        if value == 0:
            run = chunk[i]
            i += 1
            for _ in range(run):
                if pos >= pixel_count:
                    break
                out[pos] = 0
                pos += 1
        else:
            out[pos] = value
            pos += 1
    if pos != pixel_count:
        raise SprParseError(f"RLE decode size mismatch: got {pos}, expected {pixel_count}")
    return bytes(out)


def indices_to_rgba(indices: bytes, width: int, height: int, palette: List[SprPaletteColor]) -> np.ndarray:
    """Convert palette indices to BGRA uint8 image."""
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    for y in range(height):
        row = y * width
        for x in range(width):
            index = indices[row + x]
            color = palette[index]
            rgba[y, x, 0] = color.blue
            rgba[y, x, 1] = color.green
            rgba[y, x, 2] = color.red
            rgba[y, x, 3] = 0 if index == 0 else 255
    return rgba


def abgr_to_bgra(raw: bytes, width: int, height: int) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
    bgra = arr.copy()
    # SPR RGBA segment stores pixels as ABGR.
    bgra[:, :, 0] = arr[:, :, 3]
    bgra[:, :, 1] = arr[:, :, 2]
    bgra[:, :, 2] = arr[:, :, 1]
    bgra[:, :, 3] = arr[:, :, 0]
    return bgra


class SprReader:
    def __init__(self, spr_path: Path):
        self.spr_path = Path(spr_path)

    def load(self) -> SprFile:
        if not self.spr_path.is_file():
            raise SprParseError(f"SPR file not found: {self.spr_path}")

        reader = BinaryReader(self.spr_path.read_bytes())
        data = reader.data
        signature = reader.read_bytes(2)
        if signature != b"SP":
            raise SprParseError(f"invalid SPR signature: {signature!r}")

        version = reader.read_uint16()
        if version not in (0x100, 0x101, 0x200, 0x201):
            raise SprParseError(f"unsupported SPR version: 0x{version:04X}")

        if version < 0x200:
            pal_count = reader.read_uint16()
            rgba_count = 0
        else:
            pal_count = reader.read_uint16()
            rgba_count = reader.read_uint16()

        palette = self._parse_palette(data[-1024:])
        indexed_frames: List[Optional[SprFrame]] = []

        for index in range(pal_count):
            frame = self._read_indexed_frame(reader, version, palette, index)
            indexed_frames.append(frame)

        rgba_frames: List[SprFrame] = []
        for index in range(rgba_count):
            rgba_frames.append(self._read_rgba_frame(reader, len(indexed_frames) + index))

        if reader.offset != len(data) - 1024:
            raise SprParseError(
                f"SPR layout mismatch at offset {reader.offset}, expected {len(data) - 1024}"
            )

        return SprFile(
            path=self.spr_path,
            version=version,
            indexed_frames=indexed_frames,
            rgba_frames=rgba_frames,
            palette=palette,
        )

    def _parse_palette(self, raw: bytes) -> List[SprPaletteColor]:
        if len(raw) != 1024:
            raise SprParseError("palette must be 1024 bytes")
        palette: List[SprPaletteColor] = []
        for i in range(256):
            base = i * 4
            palette.append(
                SprPaletteColor(
                    red=raw[base],
                    green=raw[base + 1],
                    blue=raw[base + 2],
                    alpha=raw[base + 3],
                )
            )
        return palette

    def _read_indexed_frame(
        self,
        reader: BinaryReader,
        version: int,
        palette: List[SprPaletteColor],
        index: int,
    ) -> Optional[SprFrame]:
        width = reader.read_int16()
        height = reader.read_int16()

        if width <= 0 or height <= 0:
            reader.skip(1)
            return None

        pixel_count = width * height
        if version >= 0x201:
            encoded_size = reader.read_uint16()
            chunk = reader.read_bytes(encoded_size)
            indices = decompress_rle_indices(chunk, pixel_count)
        else:
            indices = reader.read_bytes(pixel_count)

        rgba = indices_to_rgba(indices, width, height, palette)
        return SprFrame(index=index, width=width, height=height, rgba=rgba, is_rgba=False)

    def _read_rgba_frame(self, reader: BinaryReader, index: int) -> SprFrame:
        width = reader.read_int16()
        height = reader.read_int16()
        if width <= 0 or height <= 0:
            raise SprParseError(f"invalid RGBA frame size: {width}x{height}")
        raw = reader.read_bytes(width * height * 4)
        rgba = abgr_to_bgra(raw, width, height)
        return SprFrame(index=index, width=width, height=height, rgba=rgba, is_rgba=True)
