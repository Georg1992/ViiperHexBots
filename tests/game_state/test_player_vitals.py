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
        self.assertEqual(loop._vitals.sp, 415)
        vitals.publish_sp(403, 500)
        self.assertEqual(loop._vitals.sp, 403)
        vitals.clear_sp()
        self.assertIsNone(loop._vitals.sp)

    def test_sp_sample_detects_stale_vs_fresh(self) -> None:
        import time

        vitals = PlayerVitals()
        vitals.publish_sp(100, 200)
        sp1, t1 = vitals.sp_sample()
        self.assertEqual(sp1, 100)
        sp2, t2 = vitals.sp_sample()
        self.assertEqual(t2, t1)
        time.sleep(0.002)
        vitals.publish_sp(100, 200)  # same value, new publish tick
        sp3, t3 = vitals.sp_sample()
        self.assertEqual(sp3, 100)
        self.assertGreater(t3, t1)
        time.sleep(0.002)
        vitals.publish_sp(88, 200)
        sp4, t4 = vitals.sp_sample()
        self.assertEqual(sp4, 88)
        self.assertGreater(t4, t3)


if __name__ == "__main__":
    unittest.main()
