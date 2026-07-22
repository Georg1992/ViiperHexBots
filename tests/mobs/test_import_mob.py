"""Unit tests for SPR/ACT mob import path resolution and install."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pybot.mobs.import_mob import (
    MobImportError,
    import_mob_from_paths,
    install_mob_assets,
    mob_assets_exist,
    resolve_spr_act_paths,
)


def _touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_resolve_pair_from_two_files(tmp_path: Path) -> None:
    spr = _touch(tmp_path / "Horn.spr")
    act = _touch(tmp_path / "Horn.act")
    got_spr, got_act = resolve_spr_act_paths([spr, act])
    assert got_spr == spr.resolve()
    assert got_act == act.resolve()


def test_resolve_pair_from_folder(tmp_path: Path) -> None:
    folder = tmp_path / "horn"
    spr = _touch(folder / "horn.spr")
    act = _touch(folder / "horn.act")
    got_spr, got_act = resolve_spr_act_paths([folder])
    assert got_spr == spr.resolve()
    assert got_act == act.resolve()


def test_resolve_rejects_mismatched_stems(tmp_path: Path) -> None:
    spr = _touch(tmp_path / "horn.spr")
    act = _touch(tmp_path / "poring.act")
    with pytest.raises(MobImportError, match="stems must match"):
        resolve_spr_act_paths([spr, act])


def test_resolve_rejects_missing_act_in_folder(tmp_path: Path) -> None:
    folder = tmp_path / "horn"
    _touch(folder / "horn.spr")
    with pytest.raises(MobImportError, match="missing matching ACT"):
        resolve_spr_act_paths([folder])


def test_resolve_rejects_only_spr(tmp_path: Path) -> None:
    spr = _touch(tmp_path / "horn.spr")
    with pytest.raises(MobImportError, match="exactly one"):
        resolve_spr_act_paths([spr])


def test_install_mob_assets_copies_lowercase(tmp_path: Path) -> None:
    spr = _touch(tmp_path / "src" / "Horn.spr", b"spr-data")
    act = _touch(tmp_path / "src" / "Horn.act", b"act-data")
    mobs = tmp_path / "mobs"
    with patch("pybot.mobs.import_mob.MOBS_DIR", mobs):
        stem = install_mob_assets(spr, act, overwrite=False)
        assert stem == "horn"
        assert (mobs / "horn" / "horn.spr").read_bytes() == b"spr-data"
        assert (mobs / "horn" / "horn.act").read_bytes() == b"act-data"
        assert mob_assets_exist("horn") is True


def test_install_requires_overwrite_when_exists(tmp_path: Path) -> None:
    spr = _touch(tmp_path / "src" / "horn.spr", b"a")
    act = _touch(tmp_path / "src" / "horn.act", b"b")
    mobs = tmp_path / "mobs"
    with patch("pybot.mobs.import_mob.MOBS_DIR", mobs):
        install_mob_assets(spr, act, overwrite=False)
        with pytest.raises(MobImportError, match="already exists"):
            install_mob_assets(spr, act, overwrite=False)
        install_mob_assets(spr, act, overwrite=True)


def test_import_mob_from_paths_builds(tmp_path: Path) -> None:
    spr = _touch(tmp_path / "src" / "horn.spr")
    act = _touch(tmp_path / "src" / "horn.act")
    mobs = tmp_path / "mobs"
    desc = tmp_path / "descriptors" / "horn.json"
    desc.parent.mkdir(parents=True, exist_ok=True)

    mock_descriptor = MagicMock()
    mock_builder = MagicMock()
    mock_builder.build.return_value = mock_descriptor

    def _fake_build(stem: str, force: bool = False):
        assert stem == "horn"
        assert force is True
        desc.write_text("{}")
        return mock_descriptor

    mock_builder.build.side_effect = _fake_build

    with (
        patch("pybot.mobs.import_mob.MOBS_DIR", mobs),
        patch("pybot.mobs.import_mob.DescriptorBuilder", return_value=mock_builder),
        patch("pybot.mobs.import_mob.descriptor_path", return_value=desc),
    ):
        entry = import_mob_from_paths([spr, act], overwrite=False)

    assert entry.descriptor_name == "horn"
    assert entry.asset_name == "horn"
    assert (mobs / "horn" / "horn.spr").is_file()
    mock_builder.build.assert_called_once_with("horn", force=True)
