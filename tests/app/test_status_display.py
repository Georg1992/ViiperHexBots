"""Pure status display helpers do not require a Tk root."""

from __future__ import annotations

import unittest

from pybot.app.status_display import format_pair


class StatusDisplayTests(unittest.TestCase):
    def test_format_pair(self) -> None:
        self.assertEqual(format_pair(100, 200), "100/200")
        self.assertEqual(format_pair(None, None), "—")
        self.assertEqual(format_pair(12, None), "12")
        self.assertEqual(format_pair(None, 99), "—/99")


if __name__ == "__main__":
    unittest.main()
