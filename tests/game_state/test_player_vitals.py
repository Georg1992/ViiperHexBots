"""Shared PlayerVitals publish → worker read."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pybot.game_state import PlayerVitals
from pybot.runtime.workers.attack_loop import AttackLoop


class PlayerVitalsTests(unittest.TestCase):
    def test_publish_and_clear(self) -> None:
        vitals = PlayerVitals()
        self.assertIsNone(vitals.sp)
        self.assertIsNone(vitals.sp_max)

        vitals.publish_sp(120, 200)
        self.assertEqual(vitals.sp, 120)
        self.assertEqual(vitals.sp_max, 200)
        self.assertEqual(vitals.sp_pair(), (120, 200))
        self.assertGreater(vitals.updated_ms, 0)

        vitals.clear_sp()
        self.assertIsNone(vitals.sp)
        self.assertIsNone(vitals.sp_max)

    def test_attack_loop_reads_shared_vitals(self) -> None:
        vitals = PlayerVitals()
        vitals.publish_sp(415, 500)
        loop = AttackLoop(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            vitals=vitals,
        )
        self.assertEqual(loop._read_sp(), 415)
        vitals.publish_sp(403, 500)
        self.assertEqual(loop._read_sp(), 403)
        vitals.clear_sp()
        self.assertIsNone(loop._read_sp())


if __name__ == "__main__":
    unittest.main()
