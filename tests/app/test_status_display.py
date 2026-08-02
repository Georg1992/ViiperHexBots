"""Pure status display helpers do not require a Tk root."""

from __future__ import annotations

import unittest

from pybot.app.status_display import format_pair, status_panel_numbers
from pybot.recognition.ui.status_panel import StatusPanelValues


class StatusDisplayTests(unittest.TestCase):
    def test_format_pair(self) -> None:
        self.assertEqual(format_pair(100, 200), "100/200")
        self.assertEqual(format_pair(None, None), "—")
        self.assertEqual(format_pair(12, None), "12")
        self.assertEqual(format_pair(None, 99), "—/99")

    def test_status_panel_numbers_contains_only_comparison_fields(self) -> None:
        values = StatusPanelValues(
            hp=10,
            hp_max=20,
            sp=30,
            sp_max=40,
            weight=50,
            weight_max=60,
            panel_origin=(7, 8),
        )
        self.assertEqual(status_panel_numbers(values), (10, 20, 30, 40, 50, 60))


if __name__ == "__main__":
    unittest.main()
