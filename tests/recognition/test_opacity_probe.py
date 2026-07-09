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
            "deathOpacityDecayRatio": 0.90,
            "deathOpacityConfirmTicks": 2,
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

    def test_decay_requires_consecutive_ticks(self) -> None:
        baseline = 0.60
        samples = 2
        streak = 0
        config = self._config()

        baseline, samples, streak, dead = evaluate_opacity_death(
            opacity_score=0.20,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=streak,
            config=config,
        )
        self.assertFalse(dead)
        self.assertEqual(streak, 1)

        baseline, samples, streak, dead = evaluate_opacity_death(
            opacity_score=0.55,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=streak,
            config=config,
        )
        self.assertFalse(dead)
        self.assertEqual(streak, 0)

        for score in (0.18, 0.17):
            baseline, samples, streak, dead = evaluate_opacity_death(
                opacity_score=score,
                baseline=baseline,
                baseline_samples=samples,
                decay_streak=streak,
                config=config,
            )
        self.assertTrue(dead)
        self.assertEqual(streak, 0)

    def test_ten_percent_drop_triggers_decay_streak(self) -> None:
        baseline = 0.60
        samples = 2
        streak = 0
        config = self._config()

        baseline, samples, streak, dead = evaluate_opacity_death(
            opacity_score=0.54,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=streak,
            config=config,
        )
        self.assertFalse(dead)
        self.assertEqual(streak, 1)

    def test_fifteen_percent_drop_triggers_decay_streak(self) -> None:
        baseline = 0.60
        samples = 2
        streak = 0
        config = self._config()

        baseline, samples, streak, dead = evaluate_opacity_death(
            opacity_score=0.51,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=streak,
            config=config,
        )
        self.assertFalse(dead)
        self.assertEqual(streak, 1)

    def test_small_drop_does_not_trigger_decay(self) -> None:
        baseline = 0.60
        samples = 2
        streak = 0
        config = self._config()

        baseline, samples, streak, dead = evaluate_opacity_death(
            opacity_score=0.57,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=streak,
            config=config,
        )
        self.assertFalse(dead)
        self.assertEqual(streak, 0)

    def test_weak_baseline_never_triggers_death(self) -> None:
        baseline = 0.10
        samples = 4
        streak = 0
        baseline, samples, streak, dead = evaluate_opacity_death(
            opacity_score=0.01,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=streak,
            config=self._config(),
        )
        self.assertFalse(dead)


if __name__ == "__main__":
    unittest.main()
