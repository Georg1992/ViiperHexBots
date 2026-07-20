"""Tests for mob descriptor startup ensure / rebuild gating."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pybot.mobs.catalog import _descriptor_needs_rebuild, ensure_mob_assets
from pybot.recognition.detector.descriptors.descriptor_builder import DESCRIPTOR_VERSION


class DescriptorEnsureTests(unittest.TestCase):
    def test_missing_file_needs_rebuild(self) -> None:
        self.assertTrue(_descriptor_needs_rebuild(Path("no_such_descriptor.json")))

    def test_corrupt_file_needs_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "descriptor.json"
            path.write_text("{not-json", encoding="utf-8")
            self.assertTrue(_descriptor_needs_rebuild(path))

    def test_stale_version_needs_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "descriptor.json"
            path.write_text(
                json.dumps({"version": max(0, DESCRIPTOR_VERSION - 1)}),
                encoding="utf-8",
            )
            # Load may fail on incomplete schema — either way must need rebuild.
            self.assertTrue(_descriptor_needs_rebuild(path))

    def test_ensure_reports_when_mobs_dir_missing(self) -> None:
        lines: list[str] = []
        with patch("pybot.mobs.catalog.MOBS_DIR", Path(tempfile.mkdtemp()) / "missing"):
            ensure_mob_assets(log_fn=lines.append)
        self.assertTrue(any("missing" in line.lower() for line in lines))


if __name__ == "__main__":
    unittest.main()
