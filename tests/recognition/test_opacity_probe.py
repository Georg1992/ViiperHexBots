"""Unit tests for opacity-based death detection."""

from __future__ import annotations

import unittest

from pybot.recognition.detector.tracking.opacity_probe import (
    calibrate_opacity_baseline,
    evaluate_opacity_death,
    is_opacity_calibrated,
)


class OpacityDeathProbeTests(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "deathOpacityBaselineSamples": 2,
            "deathOpacityMinBaseline": 0.20,
            "deathOpacityDropRatio": 0.80,
            "deathOpacityConfirmMs": 300,
            "deathSpNoSpendConfirmMs": 150,
        }

    def test_baseline_calibration_blocks_death(self) -> None:
        baseline = 0.0
        samples = 0
        for score in (0.55, 0.58):
            baseline, samples = calibrate_opacity_baseline(
                opacity_score=score,
                baseline=baseline,
                baseline_samples=samples,
                config=self._config(),
            )
        self.assertEqual(samples, 2)
        self.assertGreaterEqual(baseline, 0.58)
        self.assertTrue(
            is_opacity_calibrated(
                baseline=baseline,
                baseline_samples=samples,
                config=self._config(),
            )
        )

    def test_tiny_jitter_below_peak_is_not_a_fade(self) -> None:
        baseline = 0.60
        samples = 2
        config = self._config()
        # Living animation often sits a hair under the peak baseline.
        baseline, samples, since, dead = evaluate_opacity_death(
            opacity_score=0.59,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=0,
            config=config,
            now_tick=1000,
        )
        self.assertFalse(dead)
        self.assertEqual(since, 0)

    def test_meaningful_fade_requires_confirm_ms(self) -> None:
        baseline = 0.60
        samples = 2
        config = self._config()
        since = 0

        baseline, samples, since, dead = evaluate_opacity_death(
            opacity_score=0.20,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=since,
            config=config,
            now_tick=1000,
        )
        self.assertFalse(dead)
        self.assertEqual(since, 1000)

        baseline, samples, since, dead = evaluate_opacity_death(
            opacity_score=0.60,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=since,
            config=config,
            now_tick=1100,
        )
        self.assertFalse(dead)
        self.assertEqual(since, 0)

        baseline, samples, since, dead = evaluate_opacity_death(
            opacity_score=0.18,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=since,
            config=config,
            now_tick=2000,
        )
        self.assertFalse(dead)
        self.assertEqual(since, 2000)

        baseline, samples, since, dead = evaluate_opacity_death(
            opacity_score=0.17,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=since,
            config=config,
            now_tick=2299,
        )
        self.assertFalse(dead)

        baseline, samples, since, dead = evaluate_opacity_death(
            opacity_score=0.16,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=since,
            config=config,
            now_tick=2300,
        )
        self.assertTrue(dead)
        self.assertEqual(since, 0)

    def test_recovery_above_drop_ratio_resets_fade(self) -> None:
        baseline = 0.60
        samples = 2
        config = self._config()

        baseline, samples, since, dead = evaluate_opacity_death(
            opacity_score=0.40,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=0,
            config=config,
            now_tick=1000,
        )
        self.assertEqual(since, 1000)
        baseline, samples, since, dead = evaluate_opacity_death(
            opacity_score=0.55,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=since,
            config=config,
            now_tick=1200,
        )
        self.assertFalse(dead)
        self.assertEqual(since, 0)

    def test_weak_baseline_never_triggers_death(self) -> None:
        baseline = 0.10
        samples = 4
        baseline, samples, since, dead = evaluate_opacity_death(
            opacity_score=0.01,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=0,
            config=self._config(),
            now_tick=5000,
        )
        self.assertFalse(dead)

    def test_drop_while_moving_holds_fade_clock(self) -> None:
        baseline = 0.60
        samples = 2
        config = self._config()
        baseline, samples, since, dead = evaluate_opacity_death(
            opacity_score=0.40,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=0,
            config=config,
            moving=True,
            now_tick=1000,
        )
        self.assertFalse(dead)
        self.assertEqual(since, 0)

        baseline, samples, since, dead = evaluate_opacity_death(
            opacity_score=0.40,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=since,
            config=config,
            moving=False,
            now_tick=1000,
        )
        self.assertEqual(since, 1000)
        baseline, samples, since, dead = evaluate_opacity_death(
            opacity_score=0.35,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=since,
            config=config,
            moving=True,
            now_tick=2000,
        )
        self.assertFalse(dead)
        self.assertEqual(since, 1000)

    def test_moving_never_confirms_death_even_after_confirm_ms(self) -> None:
        baseline = 0.60
        samples = 2
        config = self._config()
        since = 1000
        for tick, score in ((1500, 0.10), (2000, 0.05), (3000, 0.01)):
            baseline, samples, since, dead = evaluate_opacity_death(
                opacity_score=score,
                baseline=baseline,
                baseline_samples=samples,
                decay_streak=since,
                config=config,
                moving=True,
                now_tick=tick,
            )
            self.assertFalse(dead)
            self.assertEqual(since, 1000)

    def test_death_silhouette_confirms_immediately(self) -> None:
        baseline = 0.60
        samples = 2
        config = self._config()
        # Corpse pose that beats living: one stationary frame is enough.
        baseline, samples, since, dead = evaluate_opacity_death(
            opacity_score=0.59,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=0,
            config=config,
            now_tick=1000,
            death_silhouette_hit=True,
        )
        self.assertTrue(dead)
        self.assertEqual(since, 0)

    def test_sp_no_spend_accelerates_confirm(self) -> None:
        baseline = 0.60
        samples = 2
        config = self._config()
        baseline, samples, since, dead = evaluate_opacity_death(
            opacity_score=0.20,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=0,
            config=config,
            now_tick=1000,
            sp_no_spend=True,
        )
        self.assertFalse(dead)
        baseline, samples, since, dead = evaluate_opacity_death(
            opacity_score=0.18,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=since,
            config=config,
            now_tick=1150,
            sp_no_spend=True,
        )
        self.assertTrue(dead)

    def test_sp_without_fade_does_not_confirm(self) -> None:
        baseline = 0.60
        samples = 2
        config = self._config()
        baseline, samples, since, dead = evaluate_opacity_death(
            opacity_score=0.59,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=0,
            config=config,
            now_tick=1000,
            sp_no_spend=True,
        )
        self.assertFalse(dead)
        self.assertEqual(since, 0)

    def test_death_silhouette_confirms_even_while_moving(self) -> None:
        baseline = 0.60
        samples = 2
        config = self._config()
        # Corpse pose always confirms — movement state is irrelevant.
        baseline, samples, since, dead = evaluate_opacity_death(
            opacity_score=0.59,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=0,
            config=config,
            moving=True,
            now_tick=1000,
            death_silhouette_hit=True,
        )
        self.assertTrue(dead)
        self.assertEqual(since, 0)


if __name__ == "__main__":
    unittest.main()
