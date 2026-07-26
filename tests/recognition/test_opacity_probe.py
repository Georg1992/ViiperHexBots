"""Unit tests for opacity-based death detection."""

from __future__ import annotations

import unittest

import numpy as np

from pybot.recognition.detector.detector import load_detector_config
from pybot.recognition.detector.tracking.opacity_probe import measure_opacity_score
from pybot.recognition.rules import MobTrack, apply_opacity_observation, evaluate_opacity_death


class OpacityDeathProbeTests(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "deathOpacityBaselineSamples": 4,
            "deathOpacityMinBaseline": 0.20,
            "deathOpacityDecayRatio": 0.80,
            "deathOpacityConfirmTicks": 3,
        }

    def test_detector_config_has_required_opacity_keys(self) -> None:
        config = load_detector_config()
        for key in (
            "deathOpacityBaselineSamples",
            "deathOpacityMinBaseline",
            "deathOpacityDecayRatio",
            "deathOpacityConfirmTicks",
        ):
            self.assertIn(key, config)

    def test_baseline_calibration_blocks_death(self) -> None:
        baseline = 0.0
        samples = 0
        streak = 0
        for score in (0.55, 0.58, 0.60, 0.57):
            baseline, samples, streak, dead = evaluate_opacity_death(
                opacity_score=score,
                baseline=baseline,
                baseline_samples=samples,
                decay_streak=streak,
                config=self._config(),
                moving=False,
            )
            self.assertFalse(dead)
        self.assertEqual(samples, 4)
        self.assertGreaterEqual(baseline, 0.57)

    def test_decay_requires_consecutive_stationary_ticks(self) -> None:
        baseline = 0.60
        samples = 4
        streak = 0
        config = self._config()

        baseline, samples, streak, dead = evaluate_opacity_death(
            opacity_score=0.20,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=streak,
            config=config,
            moving=False,
        )
        self.assertFalse(dead)
        self.assertEqual(streak, 1)

        baseline, samples, streak, dead = evaluate_opacity_death(
            opacity_score=0.55,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=streak,
            config=config,
            moving=False,
        )
        self.assertFalse(dead)
        self.assertEqual(streak, 0)

        for score in (0.18, 0.17, 0.16):
            baseline, samples, streak, dead = evaluate_opacity_death(
                opacity_score=score,
                baseline=baseline,
                baseline_samples=samples,
                decay_streak=streak,
                config=config,
                moving=False,
            )
        self.assertTrue(dead)
        self.assertEqual(streak, 0)

    def test_moving_holds_decay_streak(self) -> None:
        baseline = 0.60
        samples = 4
        streak = 1
        baseline, samples, streak, dead = evaluate_opacity_death(
            opacity_score=0.20,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=streak,
            config=self._config(),
            moving=True,
        )
        self.assertFalse(dead)
        self.assertEqual(streak, 1)

    def test_small_drop_does_not_trigger_decay(self) -> None:
        baseline = 0.60
        samples = 4
        streak = 0
        baseline, samples, streak, dead = evaluate_opacity_death(
            opacity_score=0.50,
            baseline=baseline,
            baseline_samples=samples,
            decay_streak=streak,
            config=self._config(),
            moving=False,
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
            moving=False,
        )
        self.assertFalse(dead)

    def test_apply_opacity_observation_mutates_track(self) -> None:
        track = MobTrack(id=1, x=100, y=100)
        config = self._config()
        for score in (0.55, 0.58, 0.60, 0.57):
            self.assertFalse(
                apply_opacity_observation(track, opacity_score=score, config=config)
            )
        self.assertEqual(track.opacity_baseline_samples, 4)
        self.assertGreaterEqual(track.opacity_baseline, 0.57)

        for score in (0.20, 0.18, 0.16):
            dead = apply_opacity_observation(track, opacity_score=score, config=config)
        self.assertTrue(dead)
        self.assertEqual(track.opacity_decay_streak, 0)

    def test_measure_opacity_empty_bbox_is_zero(self) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        class _Desc:
            body_palette = []
            match_palette_bgr = []

        score = measure_opacity_score(frame, _Desc(), (0, 0, 0, 0), 20.0, 0.22)
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
