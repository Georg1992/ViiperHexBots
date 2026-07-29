"""Unit tests: SpriteGrf load+save preserves original header bytes byte-for-byte."""

from __future__ import annotations

import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from pybot.mobs.sprite_grf import (
    SpriteGrf,
    _GRF_HEADER_SIZE,
    _GRF_HEADER_SIZE_LEGACY,
    _GRF_KEY,
    _GRF_MAGIC,
    _GRF_VERSION,
    _encode_table,
)


def _assert_header_match_up_to_table_offset(
    orig: bytes, saved: bytes, header_size: int
) -> None:
    """Assert all header bytes match, except the ``table_offset`` field."""
    tail_len = header_size - 16
    tail_off = 14 if tail_len == 30 else 15
    off_start = 16 + tail_off
    off_end = off_start + 3

    for i in range(header_size):
        if off_start <= i <= off_end:
            continue
        assert orig[i] == saved[i], (
            f"Header byte {i} differs: orig=0x{orig[i]:02X} "
            f"saved=0x{saved[i]:02X}"
        )


def _build_grf(
    path: Path,
    *,
    header_size: int,
    seed: int = 0,
    reserved: bytes | None = None,
    entries: list | None = None,
) -> None:
    """Write a minimal valid GRF with one file entry."""
    file_data = b"hello-grf-data"
    if entries is None:
        entries = [_make_entry("data\\\\sprite\\\\test.spr", 0, file_data)]
    table_raw = _encode_table(entries)
    comp_table = zlib.compress(table_raw, 9)

    comp_data = zlib.compress(file_data)

    header = bytearray()
    header += _GRF_MAGIC
    header += _GRF_KEY
    header += struct.pack("<I", len(comp_table))
    header += struct.pack("<I", seed)

    if header_size == _GRF_HEADER_SIZE:
        header += struct.pack("<I", len(entries))
        header += struct.pack("<I", _GRF_VERSION)
    else:
        header += reserved if reserved is not None else b"\x00" * 7

    table_hdr = struct.pack("<I", len(comp_table))
    table_hdr += struct.pack("<I", len(table_raw))

    path.write_bytes(bytes(header) + comp_data + table_hdr + comp_table)


def _make_entry(path: str, offset: int, data: bytes):
    """Create a _GrfEntry for use in _encode_table."""
    from pybot.mobs.sprite_grf import _GrfEntry

    return _GrfEntry(
        path=path,
        comp_size=len(data),
        aligned_size=len(data),
        uncomp_size=len(data),
        flags=0x01,
        offset=offset,
        raw_data=data,
    )


class SpriteGrfRoundtripTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.legacy_grf = self.tmp / "test_legacy.grf"
        self.standard_grf = self.tmp / "test_standard.grf"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def _build_legacy_grf(path: Path) -> None:
        _build_grf(
            path,
            header_size=_GRF_HEADER_SIZE_LEGACY,
            seed=0x0B000000,
            reserved=bytes([0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00]),
        )

    @staticmethod
    def _build_standard_grf(path: Path) -> None:
        _build_grf(path, header_size=_GRF_HEADER_SIZE, seed=0x2A000000)

    def test_legacy_header_bytes_preserved(self) -> None:
        self._build_legacy_grf(self.legacy_grf)
        orig = self.legacy_grf.read_bytes()
        header_size = _GRF_HEADER_SIZE_LEGACY

        g = SpriteGrf(self.legacy_grf)
        self.assertEqual(g._header_size, header_size)

        out_path = self.tmp / "roundtrip_legacy.grf"
        g._path = out_path
        g.save()

        saved = out_path.read_bytes()
        self.assertTrue(saved, "roundtrip file must not be empty")
        _assert_header_match_up_to_table_offset(orig, saved, header_size)

    def test_standard_header_bytes_preserved(self) -> None:
        self._build_standard_grf(self.standard_grf)
        orig = self.standard_grf.read_bytes()
        header_size = _GRF_HEADER_SIZE

        g = SpriteGrf(self.standard_grf)
        self.assertEqual(g._header_size, header_size)

        out_path = self.tmp / "roundtrip_standard.grf"
        g._path = out_path
        g.save()

        saved = out_path.read_bytes()
        self.assertTrue(saved)
        _assert_header_match_up_to_table_offset(orig, saved, header_size)

    def test_roundtrip_loads_correctly(self) -> None:
        self._build_legacy_grf(self.legacy_grf)
        g = SpriteGrf(self.legacy_grf)
        self.assertEqual(g._header_size, _GRF_HEADER_SIZE_LEGACY)

        out_path = self.tmp / "roundtrip_load.grf"
        g._path = out_path
        g.save()

        g2 = SpriteGrf(out_path)
        self.assertEqual(g2._header_size, g._header_size)
        self.assertEqual(g2._table_compressed_first, g._table_compressed_first)
        self.assertEqual(len(g2.files()), len(g.files()))

    def test_nonzero_seed_preserved(self) -> None:
        self._build_legacy_grf(self.legacy_grf)
        g = SpriteGrf(self.legacy_grf)
        out_path = self.tmp / "roundtrip_seed.grf"
        g._path = out_path
        g.save()

        g2 = SpriteGrf(out_path)
        self.assertTrue(g2._orig_header_tail, "_orig_header_tail must not be empty")
        seed = struct.unpack_from("<I", g2._orig_header_tail, 19)[0]
        self.assertEqual(seed, 0x0B000000, f"Seed should be 0x0B000000, got 0x{seed:08X}")

    def test_nonzero_reserved_preserved(self) -> None:
        self._build_legacy_grf(self.legacy_grf)
        g = SpriteGrf(self.legacy_grf)
        out_path = self.tmp / "roundtrip_reserved.grf"
        g._path = out_path
        g.save()

        g2 = SpriteGrf(out_path)
        self.assertTrue(g2._orig_header_tail, "_orig_header_tail must not be empty")
        reserved = g2._orig_header_tail[23:30]
        expected = bytes([0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00])
        self.assertEqual(reserved, expected, f"Reserved: {reserved.hex()}, expected: {expected.hex()}")

    def test_roundtrip_new_grf_defaults(self) -> None:
        g = SpriteGrf(self.tmp / "nonexistent.grf")
        g._path = self.tmp / "new.grf"
        g._header_size = _GRF_HEADER_SIZE
        g.add_file("data\\\\sprite\\\\test.spr", b"test-data")
        g.save()

        g2 = SpriteGrf(g._path)
        self.assertEqual(g2._header_size, _GRF_HEADER_SIZE)
        self.assertTrue(g2._table_compressed_first)
        self.assertEqual(len(g2.files()), 1)


if __name__ == "__main__":
    unittest.main()
