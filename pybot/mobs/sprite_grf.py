"""Sprite GRF archive — read, list, sync modified-sprite SPR+ACT files.

The ``sprite.grf`` at the project root is a Ragnarok Online GRF archive used
on servers that allow GRF modifications.  It bundles modified (big+red) SPR+ACT
files so the game client loads the transformed sprites instead of the originals.

At startup ``sync_sprite_grf()`` ensures every mob with ``modified_sprite/``
assets has its SPR+ACT pair in the archive.
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path


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

        # Auto-detect the header layout by validating the table header. The
        # old heuristic looked for zlib's 0x78 marker at byte 46, but an
        # empty archive starts with the table-size integer instead.
        candidates = (
            (_GRF_HEADER_SIZE_LEGACY, _GRF_HEADER_SIZE)
            if len(raw) > 46 and raw[46] == 0x78
            else (_GRF_HEADER_SIZE, _GRF_HEADER_SIZE_LEGACY)
        )
        for candidate in candidates:
            table_off_pos = 30 if candidate == _GRF_HEADER_SIZE_LEGACY else 31
            if len(raw) < candidate or table_off_pos + 4 > len(raw):
                continue
            table_hint = struct.unpack_from("<I", raw, table_off_pos)[0]
            table_hdr_off = candidate + table_hint
            if table_hdr_off + 8 > len(raw):
                continue
            first, second = struct.unpack_from("<II", raw, table_hdr_off)
            for comp_size, uncomp_size in ((first, second), (second, first)):
                end = table_hdr_off + 8 + comp_size
                if end > len(raw):
                    continue
                try:
                    table_data = zlib.decompress(raw[table_hdr_off + 8:end])
                except zlib.error:
                    continue
                if len(table_data) == uncomp_size:
                    self._header_size = candidate
                    break
            else:
                continue
            break
        else:
            raise ValueError("GRF table header not found")

        # Preserve the entire post-magic header region byte-for-byte.
        self._orig_header_tail = raw[16:self._header_size]

        # The table_offset in the header is relative to the end of the
        # header; _find_table_header adds header_size for absolute position.
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
        except (zlib.error, ValueError):
            # Try swapped: u2=comp, u1=uncomp
            comp_table_size, uncomp_table_size = u2, u1
            comp_table = raw[table_hdr_off + 8 : table_hdr_off + 8 + comp_table_size]
            try:
                table_data = zlib.decompress(comp_table)
            except zlib.error as e:
                raise ValueError(
                    f"Failed to decompress GRF table (both layouts): {e}"
                ) from e
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

    def _find_table_header(self, table_hint: int) -> int:
        """Convert relative table_offset from header to absolute file position."""
        return self._header_size + table_hint

    # ------------------------------------------------------------------
    #  Query
    # ------------------------------------------------------------------

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

    def remove_entries_by_path(self, path_bytes: bytes) -> int:
        """Remove every entry whose raw archive path exactly matches *path_bytes*."""
        kept = [entry for entry in self._entries if entry._path_bytes != path_bytes]
        removed = len(self._entries) - len(kept)
        if removed:
            self._entries = kept
        return removed

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

        The archive is written to a temporary file in the same directory and
        atomically replaced into place. This keeps the production asset valid
        if the process is interrupted while regenerating it.
        """
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

        rel_table_offset = len(container)
        
        if self._orig_header_tail:
            # Modify the existing header tail to update the table offset.
            # In legacy GRFs (46-byte header), the tail is 30 bytes and offset is at 14:18.
            # In standard GRFs (47-byte header), the tail is 31 bytes and offset is at 15:19.
            tail = bytearray(self._orig_header_tail)
            off_start = 14 if len(tail) == 30 else 15
            tail[off_start:off_start+4] = struct.pack("<I", rel_table_offset)
            header += bytes(tail)
        else:
            # New GRF: write the real offset, and use the legacy 46-byte header format.
            # Legacy key is 14 bytes: 0x01 .. 0x0E
            legacy_key = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
            header += legacy_key
            
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
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self._path.name}.",
            suffix=".tmp",
            dir=str(self._path.parent),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with open(tmp_path, "wb") as f:
                f.write(bytes(header))
                f.write(bytes(container))
                f.write(table_hdr)
                f.write(comp_table)
                f.write(b"\x00" * 112)
            os.replace(tmp_path, self._path)
        finally:
            tmp_path.unlink(missing_ok=True)


# ------------------------------------------------------------------
#  Public helpers
# ------------------------------------------------------------------


def _empty_sprite_grf(path: Path) -> SpriteGrf:
    """Create an uninitialized empty archive targeting *path*."""
    grf = SpriteGrf.__new__(SpriteGrf)
    grf._path = path
    grf._entries = []
    grf._data_section = b""
    grf._header_size = _GRF_HEADER_SIZE
    grf._orig_header_tail = b""
    grf._raw_container = b""
    return grf


def remove_mob_from_sprite_grf(
    project_root: Path | None,
    mob_name: str,
) -> int:
    """Remove all GRF sprite entries belonging to one mob.

    The archive may contain duplicate entries from older syncs, so every
    matching SPR/ACT entry is removed before the archive is saved.
    """
    root = project_root or Path.cwd()
    grf_path = root / "sprite.grf"
    if not grf_path.is_file():
        # Keep the production asset present even when there is nothing to
        # remove. A later mob import can then update this archive in place.
        _empty_sprite_grf(grf_path).save()
        return 0

    grf = SpriteGrf(grf_path)
    stem = mob_name.strip().lower()
    removed = 0
    for extension in ("spr", "act"):
        path_bytes = _GRF_SPRITE_DIR_BYTES + b"\\" + f"{stem}.{extension}".encode(
            "utf-8"
        )
        removed += grf.remove_entries_by_path(path_bytes)

    if removed:
        grf.save()
    return removed


def sync_sprite_grf(
    project_root: Path | None = None,
    *,
    mob_name: str | None = None,
    remove_mob_name: str | None = None,
    logger: object = None,
) -> int:
    """Ensure modified-sprite assets are synced into ``sprite.grf``.

    If ``sprite.grf`` does not exist a fresh archive is created.
    For each mob that has a ``modified_sprite/``
    folder, any existing entry with the same filename is removed and
    replaced with the modified SPR+ACT pair. When ``remove_mob_name`` is
    supplied, that mob's entries are removed during the same archive
    regeneration and its source files are skipped.

    Called at bot startup (from ``ensure_mob_assets``) and after mob asset
    changes. Returns the
    number of files added or replaced.
    """
    from pybot.paths import MOBS_DIR

    root = project_root or Path.cwd()
    grf_path = root / "sprite.grf"

    archive_needs_write = not grf_path.is_file()
    if grf_path.is_file():
        try:
            grf = SpriteGrf(grf_path)
        except (ValueError, zlib.error, struct.error):
            # Rebuild a corrupt archive from the current modified assets.
            grf = _empty_sprite_grf(grf_path)
            archive_needs_write = True
    else:
        grf = _empty_sprite_grf(grf_path)

    def _log(msg: str) -> None:
        if logger and hasattr(logger, "__call__"):
            try:
                logger(msg)
            except UnicodeEncodeError:
                # Paths can carry non-UTF-8 bytes (Korean dirs); a logger that
                # encodes to the console codepage must never abort a sync.
                pass

    changed = 0
    remove_key = remove_mob_name.strip().lower() if remove_mob_name else None
    if remove_key:
        for extension in ("spr", "act"):
            path_bytes = _GRF_SPRITE_DIR_BYTES + b"\\" + f"{remove_key}.{extension}".encode(
                "utf-8"
            )
            changed += grf.remove_entries_by_path(path_bytes)

    # Build the complete desired set during a normal sync. This makes the
    # archive authoritative after startup and repairs stale entries left by
    # removed or manually edited mob assets.
    desired: dict[bytes, bytes] = {}
    if MOBS_DIR.is_dir():
        for mob_dir in sorted(MOBS_DIR.iterdir()):
            if not mob_dir.is_dir():
                continue
            if mob_name and mob_dir.name.lower() != mob_name.lower():
                continue
            modified_dir = mob_dir / "modified_sprite"
            if not modified_dir.is_dir():
                continue

            for spr_path in sorted(modified_dir.glob("*.spr")):
                stem = spr_path.stem.lower()
                if remove_key and stem == remove_key:
                    continue
                act_src = modified_dir / f"{stem}.act"
                if not act_src.is_file():
                    continue
                desired[
                    _GRF_SPRITE_DIR_BYTES + b"\\" + f"{stem}.spr".encode("utf-8")
                ] = spr_path.read_bytes()
                desired[
                    _GRF_SPRITE_DIR_BYTES + b"\\" + f"{stem}.act".encode("utf-8")
                ] = act_src.read_bytes()

    # A full sync is authoritative: remove every managed SPR/ACT entry that
    # no longer has a corresponding modified-sprite source file. A targeted
    # sync (mob_name=...) leaves other mobs untouched.
    if mob_name is None:
        managed_prefix = _GRF_SPRITE_DIR_BYTES + b"\\"
        stale_paths = {
            path_bytes
            for entry in grf._entries
            for path_bytes in (getattr(entry, "_path_bytes", None),)
            if path_bytes is not None
            and path_bytes.startswith(managed_prefix)
            and path_bytes.rsplit(b"\\", 1)[-1].lower().endswith(
                (b".spr", b".act")
            )
            and path_bytes not in desired
        }
        for path_bytes in stale_paths:
            changed += grf.remove_entries_by_path(path_bytes)

    for path_bytes, new_data in desired.items():
        matches = [
            entry
            for entry in grf._entries
            if entry._path_bytes == path_bytes
        ]
        existing_entry = matches[0] if matches else None

        if len(matches) == 1 and existing_entry.raw_data == new_data:
            continue  # already up to date

        if matches:
            changed += grf.remove_entries_by_path(path_bytes)

        grf.add_file_raw(path_bytes, new_data)
        changed += 1
        _log(f"[GRF] {'replaced' if existing_entry else 'added'} "
             f"{path_bytes.decode('utf-8', errors='replace')}")

    if changed > 0 or archive_needs_write:
        grf.save()
        _log(f"[GRF] sprite.grf updated — {changed} file(s) changed")

    return changed
