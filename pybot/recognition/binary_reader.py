"""Little-endian binary reader for SPR/ACT parsers."""

from __future__ import annotations

import struct
from typing import Tuple


class BinaryReader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0
        self.size = len(data)

    def remaining(self) -> int:
        return self.size - self.offset

    def read_bytes(self, count: int) -> bytes:
        if self.offset + count > self.size:
            raise EOFError(f"unexpected EOF at {self.offset}, need {count} bytes")
        chunk = self.data[self.offset : self.offset + count]
        self.offset += count
        return chunk

    def read_uint8(self) -> int:
        return self.read_bytes(1)[0]

    def read_uint16(self) -> int:
        return struct.unpack_from("<H", self.read_bytes(2))[0]

    def read_int16(self) -> int:
        return struct.unpack_from("<h", self.read_bytes(2))[0]

    def read_uint32(self) -> int:
        return struct.unpack_from("<I", self.read_bytes(4))[0]

    def read_int32(self) -> int:
        return struct.unpack_from("<i", self.read_bytes(4))[0]

    def read_float(self) -> float:
        return struct.unpack_from("<f", self.read_bytes(4))[0]

    def read_fixed_string(self, length: int) -> str:
        raw = self.read_bytes(length)
        return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace")

    def skip(self, count: int) -> None:
        self.read_bytes(count)

    def peek_uint16(self) -> int:
        return struct.unpack_from("<H", self.data, self.offset)[0]
