"""Ensure discovery pipeline docs stay in sync with production detector code."""

from __future__ import annotations

import unittest

from pybot.recognition.detector.discovery_pipeline import (
    DISCOVERY_PIPELINE,
    assert_discovery_pipeline_matches_source,
    format_discovery_pipeline_text,
)


class DiscoveryPipelineDocTests(unittest.TestCase):
    def test_pipeline_matches_production_source(self) -> None:
        assert_discovery_pipeline_matches_source()

    def test_pipeline_text_covers_all_stages(self) -> None:
        text = format_discovery_pipeline_text()
        self.assertTrue(text.startswith("Discovery pipeline\n"))
        for index, stage in enumerate(DISCOVERY_PIPELINE, start=1):
            self.assertIn(f"{index}. {stage.title}", text)
            self.assertIn(f"   a) {stage.items[0]}", text)


if __name__ == "__main__":
    unittest.main()
