"""Unit tests for SPR/ACT mob import path resolution and install."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pybot.mobs.import_mob import (
    MobImportError,
    delete_mob_assets,
    import_mob_from_paths,
    install_mob_assets,
    mob_assets_exist,
    resolve_spr_act_paths,
)


def _touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class MobImportTests(unittest.TestCase):
    def test_resolve_pair_from_two_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spr = _touch(tmp_path / "Horn.spr")
            act = _touch(tmp_path / "Horn.act")
            got_spr, got_act = resolve_spr_act_paths([spr, act])
            self.assertEqual(got_spr, spr.resolve())
            self.assertEqual(got_act, act.resolve())

    def test_resolve_rejects_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            folder = tmp_path / "horn"
            _touch(folder / "horn.spr")
            _touch(folder / "horn.act")
            with self.assertRaisesRegex(MobImportError, "folders are not supported"):
                resolve_spr_act_paths([folder])

    def test_resolve_rejects_mismatched_stems(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spr = _touch(tmp_path / "horn.spr")
            act = _touch(tmp_path / "poring.act")
            with self.assertRaisesRegex(MobImportError, "stems must match"):
                resolve_spr_act_paths([spr, act])

    def test_resolve_rejects_only_spr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spr = _touch(tmp_path / "horn.spr")
            with self.assertRaisesRegex(MobImportError, "exactly one"):
                resolve_spr_act_paths([spr])

    def test_install_mob_assets_copies_lowercase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spr = _touch(tmp_path / "src" / "Horn.spr", b"spr-data")
            act = _touch(tmp_path / "src" / "Horn.act", b"act-data")
            mobs = tmp_path / "mobs"
            with patch("pybot.mobs.import_mob.MOBS_DIR", mobs):
                stem = install_mob_assets(spr, act, overwrite=False)
                self.assertEqual(stem, "horn")
                self.assertEqual((mobs / "horn" / "sprite" / "horn.spr").read_bytes(), b"spr-data")
                self.assertEqual((mobs / "horn" / "sprite" / "horn.act").read_bytes(), b"act-data")
                self.assertTrue(mob_assets_exist("horn"))

    def test_install_requires_overwrite_when_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spr = _touch(tmp_path / "src" / "horn.spr", b"a")
            act = _touch(tmp_path / "src" / "horn.act", b"b")
            mobs = tmp_path / "mobs"
            with patch("pybot.mobs.import_mob.MOBS_DIR", mobs):
                install_mob_assets(spr, act, overwrite=False)
                with self.assertRaisesRegex(MobImportError, "already exists"):
                    install_mob_assets(spr, act, overwrite=False)
                install_mob_assets(spr, act, overwrite=True)

    def test_import_restores_files_when_grf_sync_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spr = _touch(tmp_path / "src" / "horn.spr")
            act = _touch(tmp_path / "src" / "horn.act")
            mobs = tmp_path / "mobs"
            descriptors = tmp_path / "descriptors"
            desc = descriptors / "horn" / "descriptor.json"
            builder = MagicMock()

            def build(stem: str, force: bool = False):
                desc.parent.mkdir(parents=True, exist_ok=True)
                desc.write_text("{}")
                return MagicMock()

            builder.build.side_effect = build
            with (
                patch("pybot.mobs.import_mob.MOBS_DIR", mobs),
                patch("pybot.mobs.import_mob.DESCRIPTORS_DIR", descriptors),
                patch("pybot.mobs.import_mob.PROJECT_ROOT", tmp_path),
                patch("pybot.mobs.import_mob.DescriptorBuilder", return_value=builder),
                patch("pybot.mobs.import_mob.descriptor_path", return_value=desc),
                patch(
                    "pybot.mobs.sprite_grf.sync_sprite_grf",
                    side_effect=RuntimeError("cannot rebuild GRF"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "cannot rebuild GRF"):
                    import_mob_from_paths([spr, act])

            self.assertFalse((mobs / "horn").exists())
            self.assertFalse((descriptors / "horn").exists())
            self.assertFalse((tmp_path / "sprite.grf").exists())

    def test_delete_mob_assets_removes_all_local_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mob_dir = tmp_path / "mobs" / "DesertWolf"
            _touch(mob_dir / "sprite" / "desert_wolf.spr")
            _touch(mob_dir / "sprite" / "desert_wolf.act")
            _touch(mob_dir / "modified_sprite" / "desert_wolf.spr")
            descriptor_dir = tmp_path / "descriptors" / "desert_wolf"
            _touch(descriptor_dir / "descriptor.json")
            _touch(descriptor_dir / "modified_sprite_descriptor.json")

            with (
                patch("pybot.mobs.import_mob.MOBS_DIR", tmp_path / "mobs"),
                patch(
                    "pybot.mobs.import_mob.DESCRIPTORS_DIR",
                    tmp_path / "descriptors",
                ),
                patch("pybot.mobs.import_mob.PROJECT_ROOT", tmp_path),
                patch(
                    "pybot.mobs.sprite_grf.sync_sprite_grf",
                    return_value=2,
                ) as sync_grf,
            ):
                delete_mob_assets("DesertWolf", "desert_wolf")

            self.assertFalse(mob_dir.exists())
            self.assertFalse(descriptor_dir.exists())
            sync_grf.assert_called_once_with(
                tmp_path,
                remove_mob_name="desert_wolf",
            )

    def test_delete_restores_assets_and_archive_when_sync_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mob_dir = tmp_path / "mobs" / "horn"
            _touch(mob_dir / "sprite" / "horn.spr")
            descriptor_dir = tmp_path / "descriptors" / "horn"
            _touch(descriptor_dir / "descriptor.json")
            archive = tmp_path / "sprite.grf"
            archive.write_bytes(b"original archive")

            with (
                patch("pybot.mobs.import_mob.MOBS_DIR", tmp_path / "mobs"),
                patch(
                    "pybot.mobs.import_mob.DESCRIPTORS_DIR",
                    tmp_path / "descriptors",
                ),
                patch("pybot.mobs.import_mob.PROJECT_ROOT", tmp_path),
                patch(
                    "pybot.mobs.sprite_grf.sync_sprite_grf",
                    side_effect=RuntimeError("cannot rebuild GRF"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "cannot rebuild GRF"):
                    delete_mob_assets("horn", "horn")

            self.assertTrue(mob_dir.is_dir())
            self.assertTrue(descriptor_dir.is_dir())
            self.assertEqual(archive.read_bytes(), b"original archive")

    def test_delete_mob_assets_rejects_missing_mob(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with (
                patch("pybot.mobs.import_mob.MOBS_DIR", tmp_path / "mobs"),
                patch(
                    "pybot.mobs.import_mob.DESCRIPTORS_DIR",
                    tmp_path / "descriptors",
                ),
            ):
                with self.assertRaisesRegex(MobImportError, "assets not found"):
                    delete_mob_assets("missing", "missing")

    def test_import_mob_from_paths_builds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spr = _touch(tmp_path / "src" / "horn.spr")
            act = _touch(tmp_path / "src" / "horn.act")
            mobs = tmp_path / "mobs"
            desc = tmp_path / "descriptors" / "horn.json"
            desc.parent.mkdir(parents=True, exist_ok=True)

            mock_descriptor = MagicMock()
            mock_builder = MagicMock()
            mock_builder.build.return_value = mock_descriptor

            def _fake_build(stem: str, force: bool = False):
                self.assertEqual(stem, "horn")
                self.assertTrue(force)
                desc.write_text("{}")
                return mock_descriptor

            mock_builder.build.side_effect = _fake_build

            with (
                patch("pybot.mobs.import_mob.MOBS_DIR", mobs),
                patch("pybot.mobs.import_mob.DESCRIPTORS_DIR", tmp_path / "descriptors"),
                patch("pybot.mobs.import_mob.PROJECT_ROOT", tmp_path),
                patch("pybot.mobs.import_mob.DescriptorBuilder", return_value=mock_builder),
                patch("pybot.mobs.import_mob.descriptor_path", return_value=desc),
                patch("pybot.mobs.sprite_grf.sync_sprite_grf", return_value=2) as sync_grf,
            ):
                entry = import_mob_from_paths([spr, act], overwrite=False)

            self.assertEqual(entry.descriptor_name, "horn")
            self.assertEqual(entry.asset_name, "horn")
            self.assertTrue((mobs / "horn" / "sprite" / "horn.spr").is_file())
            mock_builder.build.assert_called_once_with("horn", force=True)
            sync_grf.assert_called_once_with(tmp_path)


if __name__ == "__main__":
    unittest.main()
