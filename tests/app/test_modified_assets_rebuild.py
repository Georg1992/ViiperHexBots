"""Modified-sprite asset rebuild gating.

Deleting ``assets/mobs/<Mob>/modified_sprite/`` must trigger regeneration on
the next startup even when ``modified_sprite_descriptor.json`` is up to date —
the descriptor alone is not proof the assets exist.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

from pybot.mobs.catalog import ensure_mob_assets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_HORN_DESCRIPTOR = (
    PROJECT_ROOT / "assets" / "generated_descriptors" / "horn" / "descriptor.json"
)


@unittest.skipUnless(REAL_HORN_DESCRIPTOR.is_file(), "Horn descriptor not available")
class ModifiedAssetsRebuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.mobs_dir = self.root / "mobs"
        self.descriptors_dir = self.root / "descriptors"

        sprite_dir = self.mobs_dir / "horn" / "sprite"
        sprite_dir.mkdir(parents=True)
        (sprite_dir / "horn.spr").write_bytes(b"spr-data")
        (sprite_dir / "horn.act").write_bytes(b"act-data")

        # Copy the real, fully-populated descriptor so _descriptor_needs_rebuild
        # passes for the normal descriptor (its schema is loadable).
        descriptor_dir = self.descriptors_dir / "horn"
        descriptor_dir.mkdir(parents=True)
        shutil.copyfile(REAL_HORN_DESCRIPTOR, descriptor_dir / "descriptor.json")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _patch_ensure(self, stack: ExitStack) -> None:
        stack.enter_context(patch("pybot.mobs.catalog.MOBS_DIR", self.mobs_dir))
        stack.enter_context(
            patch("pybot.mobs.catalog.DESCRIPTORS_DIR", self.descriptors_dir)
        )
        stack.enter_context(
            patch("pybot.mobs.sprite_grf.sync_sprite_grf", return_value=0)
        )

    def _patched_builder(self, stack: ExitStack) -> MagicMock:
        builder = MagicMock()
        stack.enter_context(
            patch(
                "pybot.recognition.detector.descriptors.descriptor_builder.DescriptorBuilder",
                return_value=builder,
            )
        )
        return builder

    def test_missing_modified_assets_trigger_rebuild_despite_current_descriptor(self) -> None:
        # Current-version modified-sprite descriptor exists, but the
        # modified_sprite/ SPR+ACT pair was deleted.
        shutil.copyfile(
            REAL_HORN_DESCRIPTOR,
            self.descriptors_dir / "horn" / "modified_sprite_descriptor.json",
        )

        with ExitStack() as stack:
            self._patch_ensure(stack)
            builder = self._patched_builder(stack)
            ensure_mob_assets(log_fn=lambda _msg: None)

        builder.build_modified_sprite.assert_called_once_with("horn", force=True)

    def test_empty_catalog_still_reconciles_sprite_grf(self) -> None:
        empty_mobs = self.root / "empty-mobs"
        with (
            patch("pybot.mobs.catalog.MOBS_DIR", empty_mobs),
            patch("pybot.mobs.sprite_grf.sync_sprite_grf", return_value=0) as sync_grf,
        ):
            ensure_mob_assets(log_fn=lambda _msg: None)

        sync_grf.assert_called_once_with(PROJECT_ROOT, logger=ANY)

    def test_present_modified_assets_skip_rebuild(self) -> None:
        modified_dir = self.mobs_dir / "horn" / "modified_sprite"
        modified_dir.mkdir(parents=True)
        (modified_dir / "horn.spr").write_bytes(b"static-spr")
        (modified_dir / "horn.act").write_bytes(b"static-act")
        shutil.copyfile(
            REAL_HORN_DESCRIPTOR,
            self.descriptors_dir / "horn" / "modified_sprite_descriptor.json",
        )

        with ExitStack() as stack:
            self._patch_ensure(stack)
            builder = self._patched_builder(stack)
            ensure_mob_assets(log_fn=lambda _msg: None)

        builder.build_modified_sprite.assert_not_called()


if __name__ == "__main__":
    unittest.main()
