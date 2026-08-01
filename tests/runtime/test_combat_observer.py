"""Unit tests for extracted runtime observation services."""

from __future__ import annotations

import unittest

from pybot.runtime.combat_observer import CombatObservation, CombatObserver
from pybot.runtime.death_sites import DeathSiteStore


class CombatObserverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observer = CombatObserver(max_observation_age_ms=1000)

    def test_sp_drop_is_a_hit(self) -> None:
        result = self.observer.classify_sp(
            pre_sp=100,
            post_sp=90,
            pre_observed_ms=100,
            post_observed_ms=200,
            pre_changed_ms=50,
            post_changed_ms=50,
            sample_now_ms=250,
        )
        self.assertIs(result.outcome, CombatObservation.HIT)
        self.assertIs(result.was_idle, False)

    def test_equal_fresh_sp_is_idle(self) -> None:
        result = self.observer.classify_sp(
            pre_sp=100,
            post_sp=100,
            pre_observed_ms=100,
            post_observed_ms=900,
            pre_changed_ms=50,
            post_changed_ms=50,
            sample_now_ms=1000,
        )
        self.assertIs(result.outcome, CombatObservation.IDLE)
        self.assertIs(result.was_idle, True)

    def test_unknown_evidence_does_not_invent_a_result(self) -> None:
        cases = (
            (None, 100, "sp-unread", 200),
            (100, None, "sp-unread", 200),
            (100, 110, "sp-increased", 200),
            (100, 100, "vitals-stale", 100),
        )
        for pre_sp, post_sp, reason, post_observed_ms in cases:
            with self.subTest(reason=reason):
                result = self.observer.classify_sp(
                    pre_sp=pre_sp,
                    post_sp=post_sp,
                    pre_observed_ms=100,
                    post_observed_ms=post_observed_ms,
                    pre_changed_ms=50,
                    post_changed_ms=50,
                    sample_now_ms=300,
                )
                self.assertIs(result.outcome, CombatObservation.UNKNOWN)
                self.assertEqual(result.reason, reason)
                self.assertIsNone(result.was_idle)

    def test_transient_change_and_stale_observation_are_unknown(self) -> None:
        transient = self.observer.classify_sp(
            pre_sp=100,
            post_sp=100,
            pre_observed_ms=100,
            post_observed_ms=200,
            pre_changed_ms=50,
            post_changed_ms=150,
            sample_now_ms=250,
        )
        stale = self.observer.classify_sp(
            pre_sp=100,
            post_sp=100,
            pre_observed_ms=100,
            post_observed_ms=200,
            pre_changed_ms=50,
            post_changed_ms=50,
            sample_now_ms=1301,
        )
        self.assertEqual(transient.reason, "sp-changed-during-window")
        self.assertEqual(stale.reason, "obs-stale")


class DeathSiteStoreTests(unittest.TestCase):
    def test_nearby_heat_is_absorbed_and_refreshes_site(self) -> None:
        store = DeathSiteStore(radius_px=10, cooldown_ms=100)
        store.record(100, 100, 1000)

        self.assertTrue(store.absorb_heat(106, 108, 1050))
        self.assertEqual(store.active_count(1149), 1)
        self.assertEqual(store.active_count(1151), 0)

    def test_distant_heat_is_not_absorbed(self) -> None:
        store = DeathSiteStore(radius_px=10, cooldown_ms=1000)
        store.record(100, 100, 1000)

        self.assertFalse(store.absorb_heat(111, 100, 1100))
        self.assertEqual(store.active_count(1100), 1)

    def test_clear_removes_all_sites(self) -> None:
        store = DeathSiteStore(radius_px=10, cooldown_ms=1000)
        store.record(100, 100, 1000)
        store.clear()
        self.assertEqual(store.active_count(1000), 0)


if __name__ == "__main__":
    unittest.main()
