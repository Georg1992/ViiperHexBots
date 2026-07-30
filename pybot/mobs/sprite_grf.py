"""Sprite GRF archive — read, list, sync modified-sprite SPR+ACT files.

The ``sprite.grf`` at the project root is a Ragnarok Online GRF archive used
on servers that allow GRF modifications.  It bundles modified (big+red) SPR+ACT
files so the game client loads the transformed sprites instead of the originals.

At startup ``sync_sprite_grf()`` ensures every mob with ``modified_sprite/``
assets has its SPR+ACT pair in the archive.
"""

from __future__ import annotations

import struct
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


# GRF archive constants.
_GRF_MAGIC = b"Master of Magic\x00"
_GRF_KEY = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0x28])
_GRF_HEADER_SIZE = 47  # magic(16) + key(15) + table_off(4) + seed(4) + file_count(4) + version(4)
_GRF_HEADER_SIZE_LEGACY = 46  # older custom format: reserved(7) instead of file_count+version
_GRF_VERSION = 0x200

# Subdirectory inside the GRF where custom sprites live.
# All RO monsters must be inside the Korean "몹몬스터" directory.
# \xb8\xf3\xbd\xba\xc5\xcd is EUC-KR for "몹몬스터" (monster).
_GRF_SPRITE_DIR_BYTES = b"data\\sprite\\\xb8\xf3\xbd\xba\xc5\xcd"


@dataclass
class _GrfEntry:
    """One file inside the GRF archive."""

    path: str       # archive path, e.g. data\\sprite\\horn\\horn.spr
    comp_size: int  # compressed size
    aligned_size: int  # aligned compressed size (padded to alignment)
    uncomp_size: int   # uncompressed size
    flags: int         # 0x01 = FILE
    offset: int        # byte offset in the decompressed data section
    raw_data: bytes    # uncompressed file content (populated on read)
    _path_bytes: bytes | None = None  # raw bytes from the original table
    _orig_compressed: bytes | None = None  # original zlib chunk from source GRF
    _orig_offset: int = 0  # original offset in source _raw_container
    _orig_aligned_size: int = 0  # original aligned_size from source table


def _parse_table(data: bytes) -> list[_GrfEntry]:
    """Parse the decompressed file table into entries."""
    entries: list[_GrfEntry] = []
    pos = 0
    while pos < len(data):
        # Read null-terminated path
        end = data.find(b"\x00", pos)
        if end < 0:
            break
        path_bytes = data[pos:end]
        path = path_bytes.decode("utf-8", errors="replace")
        pos = end + 1

        if pos + 17 > len(data):
            break  # incomplete entry, end of table

        comp_size = struct.unpack_from("<I", data, pos)[0]
        aligned_size = struct.unpack_from("<I", data, pos + 4)[0]
        uncomp_size = struct.unpack_from("<I", data, pos + 8)[0]
        flags = data[pos + 12]
        offset = struct.unpack_from("<I", data, pos + 13)[0]
        pos += 17

        entries.append(_GrfEntry(
            path=path,
            comp_size=comp_size,
            aligned_size=aligned_size,
            uncomp_size=uncomp_size,
            flags=flags,
            offset=offset,
            raw_data=b"",
            _path_bytes=path_bytes,
        ))
    return entries


def _encode_table(entries: list[_GrfEntry]) -> bytes:
    """Encode file table entries back to binary."""
    buf = bytearray()
    for entry in entries:
        # Use original raw path bytes when available (preserves the exact
        # encoding from the source GRF, e.g. EUC-KR for Korean paths).
        path_bytes = (
            entry._path_bytes
            if entry._path_bytes is not None
            else entry.path.encode("utf-8")
        )
        buf += path_bytes + b"\x00"
        buf += struct.pack("<I", entry.comp_size)
        buf += struct.pack("<I", entry.aligned_size)
        buf += struct.pack("<I", entry.uncomp_size)
        buf += struct.pack("<B", entry.flags)
        buf += struct.pack("<I", entry.offset)
    return bytes(buf)


def _zlib_compress_compat(data: bytes) -> bytes:
    """Compress *data* into a zlib stream compatible with old RO viewers.

    Python 3.14 ships with ``zlib-ng``, whose DEFLATE output is rejected
    by the RO client's decompressor for file-level chunks (the table
    compression using plain ``zlib.compress`` still works).

    The workaround is to use PowerShell's .NET ``DeflateStream`` which
    produces "classic" DEFLATE that the old client accepts.  We then wrap
    it in a standard zlib header (``0x78 0x9C`` — default compression,
    no preset dictionary) and append the adler32 checksum.
    """
    # Use PowerShell .NET DeflateStream, falling back to zlib.compress
    # when PowerShell is unavailable (e.g. CI, headless environments).
    try:
        return _powershell_deflate(data)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return zlib.compress(data, 9)


def _powershell_deflate(data: bytes) -> bytes:
    """Compress *data* via PowerShell .NET DeflateStream.

    Returns raw DEFLATE (without zlib wrapper).  Caller must add the
    ``0x78 0x9C`` header and adler32 trailer.
    """
    import os as _os

    # mkstemp returns (fd, pathname) — close the FD right away.
    fd1, in_path = tempfile.mkstemp(suffix=".raw")
    _os.close(fd1)
    fd2, out_path = tempfile.mkstemp(suffix=".deflate")
    _os.close(fd2)
    fd3, script_path = tempfile.mkstemp(suffix=".ps1")
    _os.close(fd3)

    tmp_in = Path(in_path)
    tmp_out = Path(out_path)
    ps_file = Path(script_path)

    # Escape single quotes in paths so PowerShell single-quoted strings
    # stay valid (double a single quote to escape it in PowerShell).
    _sq = lambda p: str(p).replace("'", "''")

    ps_script = f"""\
$in = [System.IO.File]::ReadAllBytes('{_sq(tmp_in)}')
$outStream = New-Object System.IO.MemoryStream
$deflateStream = New-Object System.IO.Compression.DeflateStream($outStream, [System.IO.Compression.CompressionMode]::Compress)
$deflateStream.Write($in, 0, $in.Length)
$deflateStream.Close()
[System.IO.File]::WriteAllBytes('{_sq(tmp_out)}', $outStream.ToArray())
"""

    try:
        tmp_in.write_bytes(data)
        ps_file.write_text(ps_script)
        
        # CREATE_NO_WINDOW prevents the console window from flashing on Windows
        creationflags = 0x08000000 if _os.name == "nt" else 0
        
        subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps_file)],
            check=True, capture_output=True, timeout=30,
            creationflags=creationflags,
        )
        raw_deflate = tmp_out.read_bytes()
    finally:
        for p in (tmp_in, tmp_out, ps_file):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    # Wrap raw DEFLATE in zlib header + adler32 trailer.
    header = b"\x78\x9c"  # CMF=0x78 (deflate, 32K window), FLG=0x9C (FLEVEL=2, FCHECK)
    adler = zlib.adler32(data) & 0xFFFFFFFF
    trailer = struct.pack(">I", adler)
    return header + raw_deflate + trailer


class SpriteGrf:
    """Read and maintain the sprite.grf archive."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else Path("sprite.grf")
        self._entries: list[_GrfEntry] = []
        self._data_section: bytes = b""  # decompressed merged file data
        self._header_size: int = _GRF_HEADER_SIZE  # actual header size (auto-detected)
        self._table_compressed_first: bool = True  # True=comp→uncomp, False=uncomp→comp
        self._orig_header_tail: bytes = b""  # raw[16:header_size] from original file
        self._raw_container: bytes = b""  # raw compressed file data (between header and table)
        if self._path.is_file():
            self._load()

    # ------------------------------------------------------------------
    #  Read
    # ------------------------------------------------------------------

    def _load(self) -> None:
        raw = self._path.read_bytes()

        # Header
        magic = raw[:16]
        if magic != _GRF_MAGIC:
            raise ValueError(f"Not a GRF archive: {self._path}")

        # Auto-detect header size: legacy 46-byte format has zlib 0x78 at
        # byte 46 (first byte of compressed data), standard 47-byte has it
        # at byte 47 (version's last byte is 0x00 for version 0x200).
        if len(raw) > 46 and raw[46] == 0x78:
            self._header_size = _GRF_HEADER_SIZE_LEGACY
        else:
            self._header_size = _GRF_HEADER_SIZE

        # Preserve the entire post-magic header region byte-for-byte.
        self._orig_header_tail = raw[16:self._header_size]

        # The table_offset in the header is the absolute byte position.
        table_off_pos = 30 if self._header_size == _GRF_HEADER_SIZE_LEGACY else 31
        table_hint = struct.unpack_from("<I", raw, table_off_pos)[0]
        table_hdr_off = self._find_table_header(table_hint)
        if table_hdr_off < self._header_size:
            raise ValueError("GRF table header not found")

        # Store the raw compressed container (individual file chunks).
        self._raw_container = raw[self._header_size:table_hdr_off]

        # Decompress table.
        u1 = struct.unpack_from("<I", raw, table_hdr_off)[0]
        u2 = struct.unpack_from("<I", raw, table_hdr_off + 4)[0]
        comp_table_size, uncomp_table_size = u1, u2
        comp_table = raw[table_hdr_off + 8 : table_hdr_off + 8 + comp_table_size]
        try:
            table_data = zlib.decompress(comp_table)
            if len(table_data) != uncomp_table_size:
                raise ValueError
            self._table_compressed_first = True
        except (zlib.error, ValueError):
            # Try swapped: u2=comp, u1=uncomp
            self._table_compressed_first = False
            comp_table_size, uncomp_table_size = u2, u1
            comp_table = raw[table_hdr_off + 8 : table_hdr_off + 8 + comp_table_size]
            table_data = zlib.decompress(comp_table)
            if len(table_data) != uncomp_table_size:
                raise ValueError(
                    f"Table size mismatch: got {len(table_data)}, "
                    f"expected {uncomp_table_size}"
                )

        self._entries = _parse_table(table_data)

        # Extract each file's raw (decompressed) data from the container.
        # The reference GRF stores files as individually-compressed zlib
        # chunks concatenated in the container.  We decompress each one
        # independently and store the raw bytes in a flat _data_section.
        self._data_section = b""
        for entry in self._entries:
            chunk = self._raw_container[entry.offset:entry.offset + entry.comp_size]
            if entry.comp_size != entry.uncomp_size:
                # Individually compressed — decompress with zlib.
                try:
                    raw_data = zlib.decompress(chunk)
                except zlib.error as e:
                    raise ValueError(
                        f"Failed to decompress {entry.path}: {e}"
                    )
            else:
                raw_data = chunk
            if len(raw_data) != entry.uncomp_size:
                raise ValueError(
                    f"Decompressed size mismatch for {entry.path}: "
                    f"got {len(raw_data)}, expected {entry.uncomp_size}"
                )
            entry.raw_data = raw_data
            # Preserve original container fields before overwriting them.
            if chunk != raw_data:  # only store when actually compressed
                entry._orig_compressed = chunk
                entry._orig_offset = entry.offset
                entry._orig_aligned_size = entry.aligned_size
            # Update offset to point into the flat _data_section.
            entry.offset = len(self._data_section)
            entry.comp_size = entry.uncomp_size
            entry.aligned_size = entry.uncomp_size
            self._data_section += raw_data

    @staticmethod
    def _find_table_header(table_hint: int) -> int:
        """Return the absolute byte position of the file table header."""
        return table_hint

    # ------------------------------------------------------------------
    #  Query
    # ------------------------------------------------------------------

    def files(self) -> list[str]:
        """Return all file paths in the archive."""
        return [e.path for e in self._entries]

    def has_file(self, path: str) -> bool:
        """Check whether *path* exists in the archive."""
        return any(e.path == path for e in self._entries)

    def has_file_by_name(self, filename: str) -> bool:
        """Check whether any entry's path ends with *filename*.

        This allows detecting mobs that are already in the GRF under a
        different folder name (e.g. Korean ``바이올렛`` for wild_rose).
        """
        # Match the trailing component after the last backslash.
        return any(e.path.rsplit("\\", 1)[-1] == filename for e in self._entries)

    # ------------------------------------------------------------------
    #  Write
    # ------------------------------------------------------------------

    def remove_entry_by_name(self, filename: str) -> None:
        """Remove the first entry whose path ends with *filename*.

        This allows replacing a mob's old SPR/ACT (e.g. wild_rose.spr)
        with a new modified version.  Shadow files are left untouched.
        """
        for i, entry in enumerate(self._entries):
            if entry.path.rsplit("\\", 1)[-1] == filename:
                self._entries.pop(i)
                return

    def add_file(self, path: str, file_data: bytes) -> None:
        """Add a file to the archive (existing paths are skipped).

        Callers must check ``has_file()`` before calling — this method
        appends new entries only.

        Files are stored raw locally; the ``save()`` method handles
        individual zlib compression when writing to the GRF.
        """
        offset = len(self._data_section)

        entry = _GrfEntry(
            path=path,
            comp_size=len(file_data),
            aligned_size=len(file_data),
            uncomp_size=len(file_data),
            flags=0x01,
            offset=offset,
            raw_data=file_data,
        )

        self._data_section += file_data
        self._entries.append(entry)

    def add_file_raw(self, path_bytes: bytes, file_data: bytes) -> None:
        """Add a file with raw path bytes (preserves encoding like EUC-KR).

        Use this when the path contains non-UTF-8 bytes (e.g. Korean folder
        names from the legacy GRF) that would be corrupted by a string path.
        """
        offset = len(self._data_section)
        path = path_bytes.decode("utf-8", errors="replace")

        entry = _GrfEntry(
            path=path,
            comp_size=len(file_data),
            aligned_size=len(file_data),
            uncomp_size=len(file_data),
            flags=0x01,
            offset=offset,
            raw_data=file_data,
            _path_bytes=path_bytes,
        )

        self._data_section += file_data
        self._entries.append(entry)

    def save(self) -> None:
        """Write the GRF archive back to disk.

        Files are stored as **individually zlib-compressed** chunks in the
        data section, matching the reference GRF format that the RO client
        expects.  Each entry's ``offset`` points to its compressed chunk
        within the raw container; ``comp_size`` is the compressed size;
        ``aligned_size`` is ``comp_size`` rounded up to 16 bytes.
        """
        if not self._entries:
            return  # nothing to save

        # Build the container by preserving the original bytes at their
        # original positions, then appending new entries at the end.
        # The RO viewer is sensitive to the EXACT container layout — the
        # order and position of each entry's compressed chunk must match
        # the source GRF.  Re-ordering or re-packing entries breaks it.
        existing_entries = [e for e in self._entries if e._orig_compressed]
        new_entries = [e for e in self._entries if not e._orig_compressed]

        # Start from the original container bytes.
        container = bytearray(self._raw_container)
        next_offset = len(container)

        # Assign table fields for existing entries (keep original positions).
        for entry in existing_entries:
            entry.offset = entry._orig_offset
            entry.comp_size = len(entry._orig_compressed)
            entry.aligned_size = entry._orig_aligned_size
            entry.flags = 0x01

        # Assign table fields for new entries (appended at end).
        for entry in new_entries:
            raw_data = entry.raw_data or self._data_section[
                entry.offset:entry.offset + entry.uncomp_size
            ]
            # Compress with the legacy-compatible method (PowerShell
            # .NET DeflateStream) so old RO viewers can decompress the
            # file chunk.  Python's zlib-ng output is rejected.
            comp = _zlib_compress_compat(raw_data)
            entry.offset = next_offset
            entry.comp_size = len(comp)
            aligned = ((len(comp) + 7) // 8) * 8
            entry.aligned_size = aligned
            entry.flags = 0x01
            container += comp
            pad = len(comp) % 8
            if pad:
                container += b"\x00" * (8 - pad)
            next_offset = len(container)

        # Build table (level 9 keeps it compact so it stays within the
        # RO viewer's scan range).
        table_raw = _encode_table(self._entries)
        comp_table = zlib.compress(table_raw, 9)

        # Header.
        header = bytearray()
        header += _GRF_MAGIC

        if self._orig_header_tail:
            rel_table_offset = self._header_size + len(container)
            # Modify the existing header tail to update the table offset.
            # In legacy GRFs (46-byte header), the tail is 30 bytes and offset is at 14:18.
            # In standard GRFs (47-byte header), the tail is 31 bytes and offset is at 15:19.
            tail = bytearray(self._orig_header_tail)
            off_start = 14 if len(tail) == 30 else 15
            tail[off_start:off_start+4] = struct.pack("<I", rel_table_offset)
            header += bytes(tail)
        else:
            # New GRF: legacy 46-byte header.
            legacy_key = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
            header += legacy_key

            rel_table_offset = _GRF_HEADER_SIZE_LEGACY + len(container)
            header += struct.pack("<I", rel_table_offset)
            
            # skip (0), count1 (len+7), version (0x200)
            count1 = len(self._entries) + 7
            header += struct.pack("<III", 0, count1, _GRF_VERSION)

        # Table header (compressed-first).
        table_hdr = struct.pack("<I", len(comp_table))
        table_hdr += struct.pack("<I", len(table_raw))

        # Anti-hint padding: 112 zero bytes after the compressed table
        # prevent the RO viewer's hint check from misreading the table position.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            f.write(bytes(header))
            f.write(bytes(container))
            f.write(table_hdr)
            f.write(comp_table)
            f.write(b"\x00" * 112)


# ------------------------------------------------------------------
#  Public helpers
# ------------------------------------------------------------------


def sync_sprite_grf(
    project_root: Path | None = None,
    *,
    mob_name: str | None = None,
    logger: object = None,
) -> int:
    """Ensure modified-sprite assets are synced into ``sprite.grf``.

    If ``sprite.grf`` does not exist a fresh archive is created.
    For each mob that has a ``modified_sprite/``
    folder, any existing entry with the same filename is removed and
    replaced with the modified SPR+ACT pair.

    Called at bot startup (from ``ensure_mob_assets``).  Returns the
    number of files added or replaced.
    """
    from pybot.paths import MOBS_DIR

    root = project_root or Path.cwd()
    grf_path = root / "sprite.grf"

    if grf_path.is_file():
        try:
            grf = SpriteGrf(grf_path)
        except ValueError:
            # Old-format GRF with relative offsets — rebuild from scratch.
            grf = SpriteGrf()
            grf._path = grf_path
    else:
        grf = SpriteGrf()
        grf._path = grf_path

    def _log(msg: str) -> None:
        if logger and hasattr(logger, "__call__"):
            logger(msg)

    changed = 0
    if not MOBS_DIR.is_dir():
        return 0

    for mob_dir in sorted(MOBS_DIR.iterdir()):
        if not mob_dir.is_dir():
            continue
        if mob_name and mob_dir.name.lower() != mob_name.lower():
            continue
        modified_dir = mob_dir / "modified_sprite"
        if not modified_dir.is_dir():
            continue

        for spr_path in sorted(modified_dir.glob("*.spr")):
            stem = spr_path.stem.lower()  # e.g. "alligator"
            act_src = modified_dir / f"{stem}.act"
            if not act_src.is_file():
                continue

            for src, ext in ((spr_path, "spr"), (act_src, "act")):
                filename = f"{stem}.{ext}"
                new_data = src.read_bytes()
                
                # Check if it already exists with identical data
                existing_entry = next(
                    (e for e in grf._entries if e.path.rsplit("\\", 1)[-1] == filename), 
                    None
                )
                
                if existing_entry and existing_entry.raw_data == new_data:
                    continue  # already up to date
                
                if existing_entry:
                    grf.remove_entry_by_name(filename)
                    
                path_bytes = _GRF_SPRITE_DIR_BYTES + b"\\" + filename.encode("utf-8")
                grf.add_file_raw(path_bytes, new_data)
                changed += 1
                _log(f"[GRF] {'replaced' if existing_entry else 'added'} "
                     f"{path_bytes.decode('utf-8', errors='replace')}")

    if changed > 0:
        grf.save()
        _log(f"[GRF] sprite.grf updated — {changed} file(s) changed")

    return changed
