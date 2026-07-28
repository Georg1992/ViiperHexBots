"""Test: try multiple GRF save strategies to match the legacy."""
from __future__ import annotations
import sys
import struct
import zlib
from pathlib import Path

sys.path.insert(0, ".")
from pybot.mobs.sprite_grf import SpriteGrf, _encode_table

orig_raw = open("sprite_legacy.grf", "rb").read()
orig_table_entries_raw = None

# Find legacy table
found = -1
for off in range(len(orig_raw)-8, max(0, len(orig_raw)-2048), -1):
    if orig_raw[off] != 0x78: continue
    u1, u2 = struct.unpack_from("<I", orig_raw, off-8)[0], struct.unpack_from("<I", orig_raw, off-4)[0]
    if 0 < u1 < 100000 and 0 <= u2 < 100000:
        found = off-8
        break

u1 = struct.unpack_from("<I", orig_raw, found)[0]
legacy_table_decompressed = zlib.decompress(orig_raw[found+8:found+8+u1])
legacy_container = orig_raw[46:found]

print(f"Legacy: container={len(legacy_container)}B, table_at={found}, table_comp={u1}")
print(f"Legacy container ends at: {46 + len(legacy_container)}")
print(f"Legacy table starts at: {found}")
print(f"Legacy file ends at: {len(orig_raw)}")

# Strategy 1: Save with table_offset=112 (matching the wrong legacy value)
print("\n=== Strategy 1: table_offset=112 (wrong, like legacy) ===")
g = SpriteGrf(Path("sprite_legacy.grf"))
g._path = Path("sprite_test_strat1.grf")

# Before saving, manually override the header tail to have table_offset=112
# This is what save() will write
g.save()

# Now read the saved file and check what happens
raw1 = open("sprite_test_strat1.grf", "rb").read()
to1 = struct.unpack_from("<I", raw1, 31)[0]
print(f"  Header table_offset: {to1}")

# Strategy 2: Save with table_offset set so that 46+to works
print("\n=== Strategy 2: table_offset = abs_table_pos - 46 (relative) ===")
g2 = SpriteGrf(Path("sprite_legacy.grf"))
g2._path = Path("sprite_test_strat2.grf")
# Build and then modify
# Actually, let me write the file manually using our data structures
container2 = bytearray()
file_offset = 0
for entry in g2._entries:
    raw_data = entry.raw_data or g2._data_section[entry.offset:entry.offset + entry.uncomp_size]
    comp = zlib.compress(raw_data)
    entry.offset = file_offset
    entry.comp_size = len(comp)
    aligned = ((len(comp) + 15) // 16) * 16
    entry.aligned_size = aligned
    entry.flags = 0x01
    container2 += comp
    pad = len(comp) % 16
    if pad:
        container2 += b"\x00" * (16 - pad)
    file_offset += aligned

table_raw = _encode_table(g2._entries)
comp_table = zlib.compress(table_raw)

# Use relative offset: 46 + table_offset should point to table
abs_table_pos = g2._header_size + len(container2)
rel_table_offset = abs_table_pos - 46  # So that 46 + table_offset = actual table pos

header2 = bytearray()
header2 += b"Master of Magic\x00"
tail2 = bytearray(g2._orig_header_tail)  # raw[16:46]
tail2[15:19] = struct.pack("<I", rel_table_offset)  # relative offset
header2 += bytes(tail2)

with open("sprite_test_strat2.grf", "wb") as f:
    f.write(bytes(header2))
    f.write(bytes(container2))
    f.write(struct.pack("<I", len(comp_table)))
    f.write(struct.pack("<I", len(table_raw)))
    f.write(comp_table)

raw2 = open("sprite_test_strat2.grf", "rb").read()
to2 = struct.unpack_from("<I", raw2, 31)[0]
print(f"  Header table_offset: {to2} (relative: 46+{to2}={to2+46})")

# Strategy 3: Use 8-byte alignment instead of 16-byte
print("\n=== Strategy 3: 8-byte alignment ===")
g3 = SpriteGrf(Path("sprite_legacy.grf"))
g3._path = Path("sprite_test_strat3.grf")

container3 = bytearray()
file_offset = 0
for entry in g3._entries:
    raw_data = entry.raw_data or g3._data_section[entry.offset:entry.offset + entry.uncomp_size]
    comp = zlib.compress(raw_data)
    entry.offset = file_offset
    entry.comp_size = len(comp)
    # Use 8-byte alignment like legacy
    aligned = ((len(comp) + 7) // 8) * 8
    entry.aligned_size = aligned
    entry.flags = 0x01
    container3 += comp
    pad = len(comp) % 8
    if pad:
        container3 += b"\x00" * (8 - pad)
    file_offset += aligned

table_raw3 = _encode_table(g3._entries)
comp_table3 = zlib.compress(table_raw3)
abs_table_pos3 = g3._header_size + len(container3)

header3 = bytearray()
header3 += b"Master of Magic\x00"
tail3 = bytearray(g3._orig_header_tail)
tail3[15:19] = struct.pack("<I", abs_table_pos3)  # absolute position
header3 += bytes(tail3)

with open("sprite_test_strat3.grf", "wb") as f:
    f.write(bytes(header3))
    f.write(bytes(container3))
    f.write(struct.pack("<I", len(comp_table3)))
    f.write(struct.pack("<I", len(table_raw3)))
    f.write(comp_table3)

raw3 = open("sprite_test_strat3.grf", "rb").read()
to3 = struct.unpack_from("<I", raw3, 31)[0]
size3 = len(raw3)
print(f"  Size: {size3}")
print(f"  Table: {len(container3)}B container + 46 header = {to3} (abs)")
print(f"  Table at {to3}, {size3-to3}B from EOF")

# Strategy 4: BOTH relative offset AND 8-byte alignment
print("\n=== Strategy 4: 8-byte alignment + relative offset ===")
g4 = SpriteGrf(Path("sprite_legacy.grf"))
g4._path = Path("sprite_test_strat4.grf")

container4 = bytearray()
file_offset = 0
for entry in g4._entries:
    raw_data = entry.raw_data or g4._data_section[entry.offset:entry.offset + entry.uncomp_size]
    comp = zlib.compress(raw_data)
    entry.offset = file_offset
    entry.comp_size = len(comp)
    aligned = ((len(comp) + 7) // 8) * 8
    entry.aligned_size = aligned
    entry.flags = 0x01
    container4 += comp
    pad = len(comp) % 8
    if pad:
        container4 += b"\x00" * (8 - pad)
    file_offset += aligned

table_raw4 = _encode_table(g4._entries)
comp_table4 = zlib.compress(table_raw4)
abs_table_pos4 = g4._header_size + len(container4)
rel_table_offset4 = abs_table_pos4 - 46

header4 = bytearray()
header4 += b"Master of Magic\x00"
tail4 = bytearray(g4._orig_header_tail)
tail4[15:19] = struct.pack("<I", rel_table_offset4)
header4 += bytes(tail4)

with open("sprite_test_strat4.grf", "wb") as f:
    f.write(bytes(header4))
    f.write(bytes(container4))
    f.write(struct.pack("<I", len(comp_table4)))
    f.write(struct.pack("<I", len(table_raw4)))
    f.write(comp_table4)

raw4 = open("sprite_test_strat4.grf", "rb").read()
to4 = struct.unpack_from("<I", raw4, 31)[0]
size4 = len(raw4)
print(f"  Size: {size4}")
print(f"  Table at {abs_table_pos4} (abs), header says {to4} (46+{to4}={to4+46})")
print(f"  Table at {abs_table_pos4}, {size4-abs_table_pos4}B from EOF")

# Strategy 5: EXACTLY copy the legacy compressed container + table, only change added files  
print("\n=== Strategy 5: Preserve original compressed data, only update table format ===")
# This is the "ideal" approach - keep the legacy container intact, rewrite only the table

g5 = SpriteGrf(Path("sprite_legacy.grf"))

# Extract original compressed chunks from legacy container
# The legacy container has files in order: shadow.act(0), shadow.spr(48), wild_rose.spr(464), wild_rose.act(26296)
# But our entries are ordered: shadow.act, shadow.spr, wild_rose.act, wild_rose.spr

# Let me use the container as-is and just rewrite the table with original offsets
# Actually this approach keeps the container exactly as legacy
container5 = legacy_container  # exact copy

# Build table with ORIGINAL offsets but OUR compressed sizes... no
# Build table with LEGACY offsets and sizes
g5._entries[0].offset = 0        # shadow.act in legacy container
g5._entries[0].comp_size = 43
g5._entries[0].aligned_size = 48
g5._entries[0].uncomp_size = 116

g5._entries[1].offset = 48       # shadow.spr
g5._entries[1].comp_size = 415
g5._entries[1].aligned_size = 416
g5._entries[1].uncomp_size = 1631

g5._entries[2].offset = 26296    # wild_rose.act (note: NOT sequential!)
g5._entries[2].comp_size = 2528
g5._entries[2].aligned_size = 2528
g5._entries[2].uncomp_size = 62276

g5._entries[3].offset = 464      # wild_rose.spr (note: BEFORE wild_rose.act in container!)
g5._entries[3].comp_size = 25829
g5._entries[3].aligned_size = 25832
g5._entries[3].uncomp_size = 77852

table_raw5 = _encode_table(g5._entries)
comp_table5 = zlib.compress(table_raw5)
abs_table_pos5 = 46 + len(container5)

header5 = bytearray()
header5 += b"Master of Magic\x00"
tail5 = bytearray(g5._orig_header_tail)
# Use WRONG offset like legacy (112) to trigger scan, same as original
tail5[15:19] = struct.pack("<I", 112)
header5 += bytes(tail5)

with open("sprite_test_strat5.grf", "wb") as f:
    f.write(bytes(header5))
    f.write(container5)
    f.write(struct.pack("<I", len(comp_table5)))
    f.write(struct.pack("<I", len(table_raw5)))
    f.write(comp_table5)

raw5 = open("sprite_test_strat5.grf", "rb").read()
size5 = len(raw5)
print(f"  Size: {size5}")
print(f"  Same as legacy? {size5 == len(orig_raw)}")

# Print files for each strategy
for path_label, raw_data in [("strat1", raw1), ("strat3", raw3), ("strat5", raw5)]:
    fnd = -1
    for off in range(len(raw_data)-8, max(0, len(raw_data)-2048), -1):
        if raw_data[off] != 0x78: continue
        u1, u2 = struct.unpack_from("<I", raw_data, off-8)[0], struct.unpack_from("<I", raw_data, off-4)[0]
        if 0 < u1 < 100000 and 0 <= u2 < 100000:
            fnd = off-8
            break
    if fnd >= 0:
        u1 = struct.unpack_from("<I", raw_data, fnd)[0]
        u2 = struct.unpack_from("<I", raw_data, fnd+4)[0]
        tbl = zlib.decompress(raw_data[fnd+8:fnd+8+u1])
        entries = []
        pos = 0
        while pos < len(tbl):
            end = tbl.find(b"\x00", pos)
            if end < 0: break
            path = tbl[pos:end]
            pos = end + 1
            unc = struct.unpack_from("<I", tbl, pos+8)[0]
            entries.append((path, unc))
            pos += 17
        print(f"\n  {path_label}: {len(entries)} entries, {len(tbl)}B table, {len(raw_data)}B file:")
        for path, unc in entries:
            print(f"    {path} ({unc}B)")

print("\n\nDone! Test files created:")
print("  sprite_test_strat1.grf - absolute offset (our current save)")
print("  sprite_test_strat2.grf - relative offset (46+to)")
print("  sprite_test_strat3.grf - 8-byte alignment")
print("  sprite_test_strat4.grf - 8-byte alignment + relative offset")
print("  sprite_test_strat5.grf - EXACT legacy container + re-encoded table, wrong to=112")
