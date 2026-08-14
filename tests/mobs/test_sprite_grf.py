"""Tests for removing mob-owned sprite.grf entries."""

from __future__ import annotations

import shutil
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pybot.mobs.sprite_grf import (
    _GRF_SPRITE_DIR_BYTES,
    SpriteGrf,
    remove_mob_from_sprite_grf,
    sync_sprite_grf,
)
from pybot.recognition.spr_reader import SprReader


class SpriteGrfRemovalTests(unittest.TestCase):
    def test_thara_modified_sprite_preserves_opaque_red_pixels(self) -> None:
        root = Path(__file__).resolve().parents[2]
        modified = root / "assets" / "mobs" / "thara_frog" / "modified_sprite"
        frame = SprReader(modified / "thara_frog.spr").load().get_frame(0)
        self.assertIsNotNone(frame)
        assert frame is not None
        opaque = frame.rgba[:, :, 3] >= 128
        self.assertTrue(opaque.any())
        self.assertGreater(int(frame.rgba[:, :, 2][opaque].min()), 0)

    def test_thara_entries_match_generated_modified_assets(self) -> None:
        root = Path(__file__).resolve().parents[2]
        archive = SpriteGrf(root / "sprite.grf")
        modified = root / "assets" / "mobs" / "thara_frog" / "modified_sprite"
        for extension in ("spr", "act"):
            path_bytes = _GRF_SPRITE_DIR_BYTES + b"\\thara_frog." + extension.encode()
            matches = [
                entry for entry in archive._entries if entry._path_bytes == path_bytes
            ]
            self.assertEqual(len(matches), 1, path_bytes)
            self.assertEqual(matches[0].raw_data, (modified / f"thara_frog.{extension}").read_bytes())

    def test_round_trips_production_archive(self) -> None:
        source = Path(__file__).resolve().parents[2] / "sprite.grf"
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "sprite.grf"
            shutil.copyfile(source, archive_path)
            original = SpriteGrf(archive_path)
            original_contents = {
                entry._path_bytes: entry.raw_data for entry in original._entries
            }

            original.save()
            round_tripped = SpriteGrf(archive_path)
            round_trip_contents = {
                entry._path_bytes: entry.raw_data
                for entry in round_tripped._entries
            }

            self.assertEqual(round_trip_contents, original_contents)

    def test_removes_all_mob_entries_and_saves_remaining_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grf_path = root / "sprite.grf"
            grf_path.write_bytes(b"archive")
            grf = SimpleNamespace(
                _entries=[
                    SimpleNamespace(path=r"data\sprite\mobmonsters\horn.spr"),
                    SimpleNamespace(path=r"data\sprite\mobmonsters\horn.act"),
                    SimpleNamespace(path=r"data\sprite\mobmonsters\other.spr"),
                ],
                save=lambda: None,
            )

            def remove(filename: str) -> None:
                for index, entry in enumerate(grf._entries):
                    if entry.path.rsplit("\\", 1)[-1] == filename:
                        grf._entries.pop(index)
                        return

            def remove_path(path_bytes: bytes) -> int:
                before = len(grf._entries)
                filename = path_bytes.rsplit(b"\\", 1)[-1].decode()
                while any(
                    entry.path.rsplit("\\", 1)[-1] == filename
                    for entry in grf._entries
                ):
                    remove(filename)
                return before - len(grf._entries)

            grf.remove_entries_by_path = remove_path
            with patch("pybot.mobs.sprite_grf.SpriteGrf", return_value=grf):
                with patch.object(grf, "save", wraps=grf.save) as save:
                    removed = remove_mob_from_sprite_grf(root, "horn")

            self.assertEqual(removed, 2)
            self.assertEqual(len(grf._entries), 1)
            save.assert_called_once()
            self.assertTrue(grf_path.is_file())

    def test_keeps_archive_after_removing_last_mob(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grf_path = root / "sprite.grf"
            grf_path.write_bytes(b"archive")
            grf = SimpleNamespace(
                _entries=[
                    SimpleNamespace(path=r"data\sprite\mobmonsters\horn.spr"),
                    SimpleNamespace(path=r"data\sprite\mobmonsters\horn.act"),
                ],
                save=lambda: None,
            )

            def remove(filename: str) -> None:
                for index, entry in enumerate(grf._entries):
                    if entry.path.rsplit("\\", 1)[-1] == filename:
                        grf._entries.pop(index)
                        return

            def remove_path(path_bytes: bytes) -> int:
                before = len(grf._entries)
                filename = path_bytes.rsplit(b"\\", 1)[-1].decode()
                while any(
                    entry.path.rsplit("\\", 1)[-1] == filename
                    for entry in grf._entries
                ):
                    remove(filename)
                return before - len(grf._entries)

            grf.remove_entries_by_path = remove_path
            with patch("pybot.mobs.sprite_grf.SpriteGrf", return_value=grf):
                removed = remove_mob_from_sprite_grf(root, "horn")

            self.assertEqual(removed, 2)
            self.assertTrue(grf_path.exists())

    def test_remove_entries_by_path_preserves_same_basename_elsewhere(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = _GRF_SPRITE_DIR_BYTES + b"\\horn.spr"
            unrelated = b"data\\sprite\\other\\horn.spr"
            grf = SpriteGrf(Path(tmp) / "sprite.grf")
            grf._entries = [
                SimpleNamespace(_path_bytes=target),
                SimpleNamespace(_path_bytes=unrelated),
            ]

            removed = grf.remove_entries_by_path(target)

            self.assertEqual(removed, 1)
            self.assertEqual(grf._entries[0]._path_bytes, unrelated)

    def test_sync_updates_archive_from_modified_assets_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mobs = root / "mobs"
            modified = mobs / "horn" / "modified_sprite"
            def write_asset(path: Path, data: bytes) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)

            write_asset(modified / "horn.spr", b"new spr")
            write_asset(modified / "horn.act", b"new act")

            target = _GRF_SPRITE_DIR_BYTES + b"\\horn.spr"
            unrelated = b"data\\sprite\\other\\horn.spr"
            archive = SpriteGrf(root / "sprite.grf")
            archive.add_file_raw(target, b"old spr")
            archive.add_file_raw(unrelated, b"keep me")
            with patch(
                "pybot.mobs.sprite_grf._zlib_compress_compat",
                side_effect=lambda data: zlib.compress(data, 9),
            ):
                archive.save()
                with patch("pybot.paths.MOBS_DIR", mobs):
                    changed = sync_sprite_grf(root)

            self.assertEqual(changed, 3)
            loaded = SpriteGrf(root / "sprite.grf")
            contents = {
                entry._path_bytes: entry.raw_data for entry in loaded._entries
            }
            self.assertEqual(contents[target], b"new spr")
            self.assertEqual(contents[_GRF_SPRITE_DIR_BYTES + b"\\horn.act"], b"new act")
            self.assertEqual(contents[unrelated], b"keep me")

    def test_sync_can_remove_mob_during_regeneration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grf_path = root / "sprite.grf"
            grf_path.write_bytes(b"archive")
            grf = SimpleNamespace(
                _entries=[
                    SimpleNamespace(path=r"data\sprite\mobmonsters\horn.spr"),
                    SimpleNamespace(path=r"data\sprite\mobmonsters\horn.act"),
                    SimpleNamespace(path=r"data\sprite\mobmonsters\other.spr"),
                ],
                save=lambda: None,
            )

            def remove(filename: str) -> None:
                for index, entry in enumerate(grf._entries):
                    if entry.path.rsplit("\\", 1)[-1] == filename:
                        grf._entries.pop(index)
                        return

            def remove_path(path_bytes: bytes) -> int:
                before = len(grf._entries)
                filename = path_bytes.rsplit(b"\\", 1)[-1].decode()
                while any(
                    entry.path.rsplit("\\", 1)[-1] == filename
                    for entry in grf._entries
                ):
                    remove(filename)
                return before - len(grf._entries)

            grf.remove_entries_by_path = remove_path
            with (
                patch("pybot.mobs.sprite_grf.SpriteGrf", return_value=grf),
                patch("pybot.paths.MOBS_DIR", root / "mobs"),
                patch.object(grf, "save", wraps=grf.save) as save,
            ):
                changed = sync_sprite_grf(root, remove_mob_name="horn")

            self.assertEqual(changed, 2)
            self.assertEqual(
                [entry.path for entry in grf._entries],
                [r"data\sprite\mobmonsters\other.spr"],
            )
            save.assert_called_once()

    def test_sync_keeps_archive_when_removed_mob_was_last_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grf_path = root / "sprite.grf"
            grf_path.write_bytes(b"archive")
            grf = SimpleNamespace(
                _entries=[
                    SimpleNamespace(path=r"data\sprite\mobmonsters\horn.spr"),
                    SimpleNamespace(path=r"data\sprite\mobmonsters\horn.act"),
                ],
                save=lambda: None,
            )

            def remove(filename: str) -> None:
                for index, entry in enumerate(grf._entries):
                    if entry.path.rsplit("\\", 1)[-1] == filename:
                        grf._entries.pop(index)
                        return

            def remove_path(path_bytes: bytes) -> int:
                before = len(grf._entries)
                filename = path_bytes.rsplit(b"\\", 1)[-1].decode()
                while any(
                    entry.path.rsplit("\\", 1)[-1] == filename
                    for entry in grf._entries
                ):
                    remove(filename)
                return before - len(grf._entries)

            grf.remove_entries_by_path = remove_path
            with (
                patch("pybot.mobs.sprite_grf.SpriteGrf", return_value=grf),
                patch("pybot.paths.MOBS_DIR", root / "mobs"),
            ):
                changed = sync_sprite_grf(root, remove_mob_name="horn")

            self.assertEqual(changed, 2)
            self.assertTrue(grf_path.exists())

    def test_sync_creates_empty_archive_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("pybot.paths.MOBS_DIR", root / "mobs"):
                changed = sync_sprite_grf(root)

            self.assertEqual(changed, 0)
            self.assertTrue((root / "sprite.grf").is_file())
            self.assertEqual(len(SpriteGrf(root / "sprite.grf")._entries), 0)

    def test_full_sync_removes_stale_mob_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = SpriteGrf(root / "sprite.grf")
            stale = _GRF_SPRITE_DIR_BYTES + b"\\stale.spr"
            keep = b"data\\sprite\\other\\keep.spr"
            archive.add_file_raw(stale, b"stale")
            archive.add_file_raw(keep, b"keep")
            with patch(
                "pybot.mobs.sprite_grf._zlib_compress_compat",
                side_effect=lambda data: zlib.compress(data, 9),
            ):
                archive.save()
                with patch("pybot.paths.MOBS_DIR", root / "mobs"):
                    changed = sync_sprite_grf(root)

            self.assertEqual(changed, 1)
            loaded = SpriteGrf(root / "sprite.grf")
            paths = {entry._path_bytes for entry in loaded._entries}
            self.assertNotIn(stale, paths)
            self.assertIn(keep, paths)


if __name__ == "__main__":
    unittest.main()
