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
        self.assertGreater(vitals.observed_ms, 0)
        self.assertEqual(vitals.changed_ms, vitals.observed_ms)

        vitals.clear_sp()
        self.assertIsNone(vitals.sp)
        self.assertIsNone(vitals.sp_max)

    def test_attack_loop_reads_shared_vitals(self) -> None:
        vitals = PlayerVitals()
        vitals.publish_sp(415, 500)
        loop = AttackLoop(
            MagicMock(character_screen_pos=MagicMock(return_value=None)),
            MagicMock(),
            MagicMock(),
            vitals=vitals,
        )
        self.assertEqual(loop._vitals.sp, 415)
        vitals.publish_sp(403, 500)
        self.assertEqual(loop._vitals.sp, 403)
        vitals.clear_sp()
        self.assertIsNone(loop._vitals.sp)

    def test_sp_sample_detects_observe_vs_change(self) -> None:
        import time

        vitals = PlayerVitals()
        vitals.publish_sp(100, 200)
        sp1, obs1, chg1 = vitals.sp_sample()
        self.assertEqual(sp1, 100)
        self.assertEqual(obs1, chg1)

        sp2, obs2, chg2 = vitals.sp_sample()
        self.assertEqual(obs2, obs1)
        self.assertEqual(chg2, chg1)

        time.sleep(0.002)
        vitals.publish_sp(100, 200)  # same value — observe bumps, change does not
        sp3, obs3, chg3 = vitals.sp_sample()
        self.assertEqual(sp3, 100)
        self.assertGreater(obs3, obs1)
        self.assertEqual(chg3, chg1)

        time.sleep(0.002)
        vitals.publish_sp(88, 200)
        sp4, obs4, chg4 = vitals.sp_sample()
        self.assertEqual(sp4, 88)
        self.assertGreater(obs4, obs3)
        self.assertGreater(chg4, chg1)
        self.assertEqual(chg4, obs4)

    def test_pre_teleport_epoch_cannot_publish_after_reset(self) -> None:
        vitals = PlayerVitals()
        old_epoch = vitals.observation_epoch
        vitals.publish_sp(574, 1454)

        vitals.begin_observation_epoch()
        self.assertIsNone(vitals.sp)
        self.assertFalse(vitals.publish_sp_if_current(574, 1454, old_epoch))
        self.assertIsNone(vitals.sp)

        current_epoch = vitals.observation_epoch
        # Teleport keeps the producer alive but quarantines transition-frame
        # reads until the landing boundary is complete.
        self.assertFalse(vitals.publish_snapshot_if_current(
            90, 100, 350, 1454, 20, 100, current_epoch,
        ))
        self.assertTrue(vitals.complete_observation_epoch(current_epoch))
        self.assertTrue(vitals.publish_snapshot_if_current(
            90, 100, 350, 1454, 20, 100, current_epoch,
        ))
        self.assertEqual(vitals.sp_pair(), (350, 1454))
        self.assertEqual(vitals.hp_pair(), (90, 100))

    def test_hp_and_weight_publish_do_not_bump_sp_clocks(self) -> None:
        import time

        vitals = PlayerVitals()
        vitals.publish_sp(100, 200)
        _sp, obs1, chg1 = vitals.sp_sample()
        time.sleep(0.002)
        vitals.publish_hp(50, 100)
        vitals.publish_weight(10, 100)
        _sp, obs2, chg2 = vitals.sp_sample()
        self.assertEqual(obs2, obs1)
        self.assertEqual(chg2, chg1)


if __name__ == "__main__":
    unittest.main()
