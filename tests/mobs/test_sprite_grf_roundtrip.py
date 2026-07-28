"""Unit tests: SpriteGrf load+save preserves original header bytes byte-for-byte."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

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
    """Assert all header bytes match, except the ``table_offset`` field.

    ``save()`` rewrites the table offset to reflect the new table position
    in the output file.  The exact bytes that change depend on the header
    format:

    * Legacy (46-byte header): tail is 30 bytes, offset at tail[14:18] =
      file bytes 30-33.
    * Standard (47-byte header): tail is 31 bytes, offset at tail[15:19] =
      file bytes 31-34.
    """
    # Determine which file bytes the table_offset write touches.
    # save() writes 4 bytes starting at tail[14] for legacy (tail=30)
    # or tail[15] for standard (tail=31).  tail[n] = file byte [16 + n].
    tail_len = header_size - 16  # 30 for legacy, 31 for standard
    tail_off = 14 if tail_len == 30 else 15
    off_start = 16 + tail_off  # convert tail index → file byte index
    off_end = off_start + 3  # inclusive

    for i in range(header_size):
        if off_start <= i <= off_end:
            continue  # table_offset always changes
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
    header += struct.pack("<I", len(comp_table))  # table_offset = compressed table size
    header += struct.pack("<I", seed)

    if header_size == _GRF_HEADER_SIZE:
        header += struct.pack("<I", len(entries))
        header += struct.pack("<I", _GRF_VERSION)
    else:
        header += reserved if reserved is not None else b"\x00" * 7

    # compressed-first table header
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


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def legacy_grf(tmp_path: Path) -> Path:
    """Legacy 46-byte GRF with non-zero seed and non-zero reserved bytes."""
    path = tmp_path / "test_legacy.grf"
    _build_grf(
        path,
        header_size=_GRF_HEADER_SIZE_LEGACY,
        seed=0x0B000000,
        reserved=bytes([0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00]),
    )
    return path


@pytest.fixture
def standard_grf(tmp_path: Path) -> Path:
    """Standard 47-byte GRF with non-zero seed."""
    path = tmp_path / "test_standard.grf"
    _build_grf(
        path,
        header_size=_GRF_HEADER_SIZE,
        seed=0x2A000000,
    )
    return path


# ── Tests ───────────────────────────────────────────────────────────


class TestRoundtripHeaderPreservation:
    def test_legacy_header_bytes_preserved(self, legacy_grf: Path) -> None:
        """Legacy GRF: all header bytes survive round-trip unchanged."""
        orig = legacy_grf.read_bytes()
        header_size = _GRF_HEADER_SIZE_LEGACY

        g = SpriteGrf(legacy_grf)
        assert g._header_size == header_size

        out_path = legacy_grf.parent / "roundtrip.grf"
        g._path = out_path
        g.save()

        saved = out_path.read_bytes()
        assert saved, "roundtrip file must not be empty"

        _assert_header_match_up_to_table_offset(orig, saved, header_size)

    def test_standard_header_bytes_preserved(self, standard_grf: Path) -> None:
        """Standard GRF: all header bytes survive round-trip unchanged.

        The ``table_offset`` field (bytes 31-34) is excluded from the
        comparison because ``save()`` always rewrites this value to reflect
        the new table position (which shifts due to anti-hint padding).
        """
        orig = standard_grf.read_bytes()
        header_size = _GRF_HEADER_SIZE

        g = SpriteGrf(standard_grf)
        assert g._header_size == header_size

        out_path = standard_grf.parent / "roundtrip.grf"
        g._path = out_path
        g.save()

        saved = out_path.read_bytes()
        assert saved

        _assert_header_match_up_to_table_offset(orig, saved, header_size)

    def test_roundtrip_loads_correctly(self, legacy_grf: Path) -> None:
        """Round-tripped GRF can be loaded back with correct format."""
        g = SpriteGrf(legacy_grf)
        assert g._header_size == _GRF_HEADER_SIZE_LEGACY

        out_path = legacy_grf.parent / "roundtrip.grf"
        g._path = out_path
        g.save()

        g2 = SpriteGrf(out_path)
        assert g2._header_size == g._header_size
        assert g2._table_compressed_first == g._table_compressed_first
        assert len(g2.files()) == len(g.files())

    def test_nonzero_seed_preserved(self, legacy_grf: Path) -> None:
        """Non-zero seed byte is preserved in round-trip."""
        g = SpriteGrf(legacy_grf)
        out_path = legacy_grf.parent / "roundtrip.grf"
        g._path = out_path
        g.save()

        g2 = SpriteGrf(out_path)
        assert g2._orig_header_tail, "_orig_header_tail must not be empty"
        seed = struct.unpack_from("<I", g2._orig_header_tail, 19)[0]
        assert seed == 0x0B000000, f"Seed should be 0x0B000000, got 0x{seed:08X}"

    def test_nonzero_reserved_preserved(self, legacy_grf: Path) -> None:
        """Non-zero reserved bytes are preserved in round-trip."""
        g = SpriteGrf(legacy_grf)
        out_path = legacy_grf.parent / "roundtrip.grf"
        g._path = out_path
        g.save()

        g2 = SpriteGrf(out_path)
        assert g2._orig_header_tail, "_orig_header_tail must not be empty"
        reserved = g2._orig_header_tail[23:30]  # 7 bytes at tail positions 23-29
        expected = bytes([0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00])
        assert reserved == expected, f"Reserved: {reserved.hex()}, expected: {expected.hex()}"

    def test_roundtrip_new_grf_defaults(self, tmp_path: Path) -> None:
        """New GRF created from scratch uses standard defaults."""
        # Avoid loading the real sprite.grf in CWD by passing a non-existent path.
        g = SpriteGrf(tmp_path / "nonexistent.grf")
        g._path = tmp_path / "new.grf"
        g._header_size = _GRF_HEADER_SIZE
        g.add_file("data\\\\sprite\\\\test.spr", b"test-data")
        g.save()

        g2 = SpriteGrf(g._path)
        assert g2._header_size == _GRF_HEADER_SIZE
        assert g2._table_compressed_first
        assert len(g2.files()) == 1
