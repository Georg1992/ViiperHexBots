"""Status panel HP/SP/Weight parsing tests."""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.ui.status_panel import (
    BINARIZE_THRESHOLD,
    DIGIT_MATCH_THRESHOLD,
    HP_SCAN_ZONE,
    MAX_ROW_GLYPHS,
    SP_SCAN_ZONE,
    StatusPanelValues,
    _classify_glyph,
    _load_digit_templates,
    _match_template_zncc,
    _parse_anchored,
    clear_template_cache,
    find_status_panel,
    read_status_panel,
    verify_status_panel_at,
)

FIXTURES_DIR = PROJECT_ROOT / "tests"
# (filename, hp, hp_max, sp, sp_max, weight, weight_max)
FIXTURE_CASES: tuple[tuple[str, int, int, int, int, int, int], ...] = (
    ("StatusPanel.png", 850, 3187, 1229, 1229, 587, 2630),
    ("StatusPanel2.png", 2260, 3348, 430, 430, 280, 2730),
    ("StatusPanel3.png", 1387, 3348, 430, 430, 280, 2730),
    ("StatusPanel4.png", 3348, 3348, 360, 430, 305, 2730),
    ("StatusPanel5.png", 3348, 3348, 365, 430, 305, 2730),
    ("W9.png", 3348, 3348, 430, 430, 297, 2730),
    ("W4.png", 3348, 3348, 430, 430, 294, 2730),
    ("W1.png", 3348, 3348, 430, 430, 291, 2730),
    ("RedWeight.png", 3424, 3424, 107, 435, 2457, 2730),
    # Trailing ``t`` of ``Weight`` sits in the ROI; must not become a leading 1.
    ("FalseWeight.png", 874, 3424, 249, 485, 427, 2730),
    # Short Zeny right-shifts ``464 / 2730``; ROI must still capture the max.
    ("WeightIssue.png", 3501, 3501, 370, 457, 464, 2730),
    # Full bar values — HP=40/40, SP=11/11, Weight=50/2030.
    ("StatusPanel6.png", 40, 40, 11, 11, 50, 2030),
    # Wide Zeny + Weight max 5000; panel at client origin.
    ("FalseWeight2.png", 543, 615, 124, 124, 466, 5000),
)


def _with_sp_fill_ratio(
    frame: np.ndarray,
    origin: tuple[int, int],
    fill_ratio: float,
) -> np.ndarray:
    """Keep SP digit ink; paint empty-bar gray over non-ink right of *fill_ratio*."""
    out = frame.copy()
    ox, oy = origin
    x, y, w, h = SP_SCAN_ZONE
    x0, y0 = ox + x, oy + y
    region = out[y0 : y0 + h, x0 : x0 + w]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    split = int(w * fill_ratio)
    right = region[:, split:].copy()
    right[gray[:, split:] >= BINARIZE_THRESHOLD] = (210, 210, 210)
    region[:, split:] = right
    return out


class StatusPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        clear_template_cache()

    def _load(self, name: str) -> np.ndarray:
        path = FIXTURES_DIR / name
        if not path.is_file():
            self.skipTest(f"missing fixture {path}")
        frame = cv2.imread(str(path))
        if frame is None:
            self.skipTest(f"unreadable fixture {path}")
        return frame

    def test_finds_panel_near_top_left(self) -> None:
        frame = self._load("StatusPanel.png")
        origin = find_status_panel(frame)
        self.assertIsNotNone(origin)
        assert origin is not None
        self.assertLessEqual(origin[0], 20)
        self.assertLessEqual(origin[1], 20)

    def test_find_fails_without_header(self) -> None:
        frame = self._load("StatusPanel.png")
        blanked = frame.copy()
        blanked[:30, :] = (40, 40, 40)
        self.assertIsNone(find_status_panel(blanked))

    def test_parser_honors_expired_deadline(self) -> None:
        """A live OCR budget must be able to abandon a stale frame promptly."""
        frame = self._load("StatusPanel.png")
        self.assertIsNone(
            read_status_panel(frame, deadline=time.monotonic() - 1.0)
        )

    def test_glyph_classifier_is_bounded_numpy_zncc(self) -> None:
        """Glyph matching must not enter OpenCV's native template matcher."""
        templates = _load_digit_templates()
        template = templates["8"][0]
        glyph = template.copy()
        self.assertAlmostEqual(_match_template_zncc(glyph, template), 1.0, places=5)
        character, score = _classify_glyph(glyph)
        self.assertEqual(character, "8")
        self.assertGreaterEqual(score, DIGIT_MATCH_THRESHOLD)

    def test_expired_glyph_budget_skips_template_matching(self) -> None:
        """A stale OCR frame cannot spend the whole budget on one glyph."""
        templates = _load_digit_templates()
        glyph = templates["8"][0]
        with patch(
            "pybot.recognition.ui.status_panel._match_template_zncc",
        ) as matcher:
            result = _classify_glyph(
                glyph,
                deadline=time.monotonic() - 1.0,
            )
        self.assertEqual(result, (None, -1.0))
        matcher.assert_not_called()

    def test_garbage_row_with_too_many_glyphs_bails_before_classifying(
        self,
    ) -> None:
        """A text-dense band must fail fast, not churn template matching.

        A post-teleport frame can put chat/text under the stale fixed-ROI
        anchor. With hundreds of candidate fragments, classifying each one
        is what previously burned the whole OCR read budget; the parser
        must reject the row before any template matching runs.
        """
        glyphs = [
            (x, np.full((7, 5), 255, dtype=np.uint8))
            for x in range(0, 150, 5)
        ]
        self.assertGreater(len(glyphs), MAX_ROW_GLYPHS)
        with patch(
            "pybot.recognition.ui.status_panel._glyph_components",
            return_value=glyphs,
        ), patch(
            "pybot.recognition.ui.status_panel._classify_glyph",
        ) as classify:
            started = time.monotonic()
            result = _parse_anchored(
                np.zeros((85, 138, 3), dtype=np.uint8),
                (-42, -45),
                HP_SCAN_ZONE,
                min_width=2,
                stop_at_slash=True,
            )
            elapsed = time.monotonic() - started
        self.assertIsNone(result)
        classify.assert_not_called()
        self.assertLess(elapsed, 0.05)

    def test_parser_passes_deadline_into_glyph_classifier(self) -> None:
        """The parser budget must cover each delegated glyph classification."""
        glyph = np.zeros((7, 3), dtype=np.uint8)
        deadline = time.monotonic() + 1.0
        with patch(
            "pybot.recognition.ui.status_panel._glyph_components",
            return_value=[(0, glyph)],
        ), patch(
            "pybot.recognition.ui.status_panel._classify_glyph",
            return_value=("1", 1.0),
        ) as classify:
            _parse_anchored(
                np.zeros((20, 20, 3), dtype=np.uint8),
                (0, 0),
                (0, 0, 10, 10),
                min_width=2,
                stop_at_slash=False,
                deadline=deadline,
            )
        classify.assert_called_once_with(glyph, deadline=deadline)

    def test_glyph_classifier_rejects_match_that_crosses_deadline(self) -> None:
        """A slow final template match cannot return a stale glyph result."""
        templates = _load_digit_templates()
        glyph = templates["8"][0]
        clock = iter((0.0, 0.0, 2.0, 2.0))
        with patch(
            "pybot.recognition.ui.status_panel.time.monotonic",
            side_effect=lambda: next(clock),
        ), patch(
            "pybot.recognition.ui.status_panel._match_template_zncc",
            return_value=1.0,
        ):
            result = _classify_glyph(glyph, deadline=1.0)
        self.assertEqual(result, (None, -1.0))

    def test_verify_status_panel_at_locked_origin(self) -> None:
        frame = self._load("FalseWeight2.png")
        origin = find_status_panel(frame)
        self.assertIsNotNone(origin)
        assert origin is not None
        self.assertTrue(verify_status_panel_at(frame, origin))
        self.assertFalse(verify_status_panel_at(frame, (origin[0] + 40, origin[1])))
        blanked = frame.copy()
        ox, oy = origin
        blanked[oy : oy + 20, ox : ox + 120] = (40, 40, 40)
        self.assertFalse(verify_status_panel_at(blanked, origin))

    def test_reads_hp_sp_weight_from_fixtures(self) -> None:
        for name, hp, hp_max, sp, sp_max, weight, weight_max in FIXTURE_CASES:
            with self.subTest(fixture=name):
                frame = self._load(name)
                values = read_status_panel(frame)
                self.assertIsNotNone(values, f"failed to read {name}")
                assert values is not None
                self.assertEqual(values.hp, hp)
                self.assertEqual(values.hp_max, hp_max)
                self.assertEqual(values.sp, sp)
                self.assertEqual(values.sp_max, sp_max)
                self.assertEqual(values.weight, weight)
                self.assertEqual(values.weight_max, weight_max)
                self.assertEqual(values.panel_origin, (0, 0))

    def test_currents_reuse_hp_without_hp_ocr(self) -> None:
        for name, hp, hp_max, sp, sp_max, weight, weight_max in FIXTURE_CASES:
            with self.subTest(fixture=name):
                frame = self._load(name)
                origin = find_status_panel(frame)
                self.assertIsNotNone(origin)
                assert origin is not None
                previous = StatusPanelValues(
                    hp=hp, hp_max=hp_max,
                    sp=0, sp_max=sp_max,
                    weight=0, weight_max=weight_max,
                    panel_origin=origin,
                )
                values = read_status_panel(
                    frame, origin=origin, skip_hp=True, previous=previous
                )
                self.assertIsNotNone(values)
                assert values is not None
                self.assertEqual(values.hp, hp)
                self.assertEqual(values.hp_max, hp_max)
                self.assertEqual(values.sp, sp)
                self.assertEqual(values.sp_max, sp_max)
                self.assertEqual(values.weight, weight)
                self.assertEqual(values.weight_max, weight_max)

    def test_reads_survive_origin_jitter(self) -> None:
        """Header match can land ±1–2px off; ROI padding must absorb that."""
        for name, hp, hp_max, sp, sp_max, weight, weight_max in FIXTURE_CASES:
            frame = self._load(name)
            origin = find_status_panel(frame)
            self.assertIsNotNone(origin)
            assert origin is not None
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    with self.subTest(fixture=name, dx=dx, dy=dy):
                        values = read_status_panel(
                            frame, origin=(origin[0] + dx, origin[1] + dy)
                        )
                        self.assertIsNotNone(values)
                        assert values is not None
                        self.assertEqual(values.hp, hp)
                        self.assertEqual(values.hp_max, hp_max)
                        self.assertEqual(values.sp, sp)
                        self.assertEqual(values.sp_max, sp_max)
                        self.assertEqual(values.weight, weight)
                        self.assertEqual(values.weight_max, weight_max)

    def test_sp_survives_bar_fill_change(self) -> None:
        """SP digits stay readable when the bar is full, empty, or split."""
        for name, hp, hp_max, sp, sp_max, weight, weight_max in FIXTURE_CASES:
            frame = self._load(name)
            origin = find_status_panel(frame)
            self.assertIsNotNone(origin)
            assert origin is not None
            for fill_ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
                with self.subTest(fixture=name, fill_ratio=fill_ratio):
                    altered = (
                        frame
                        if fill_ratio >= 1.0
                        else _with_sp_fill_ratio(frame, origin, fill_ratio)
                    )
                    values = read_status_panel(altered, origin=origin)
                    self.assertIsNotNone(values)
                    assert values is not None
                    self.assertEqual(values.hp, hp)
                    self.assertEqual(values.hp_max, hp_max)
                    self.assertEqual(values.sp, sp)
                    self.assertEqual(values.sp_max, sp_max)
                    self.assertEqual(values.weight, weight)
                    self.assertEqual(values.weight_max, weight_max)

    def test_sp_survives_dark_bar_fill(self) -> None:
        """Dark SP fill must not become ink — only near-black digits count."""
        frame = self._load("StatusPanel2.png")
        origin = find_status_panel(frame)
        self.assertIsNotNone(origin)
        assert origin is not None
        ox, oy = origin
        x, y, w, h = SP_SCAN_ZONE
        x0, y0 = ox + x, oy + y
        gray = cv2.cvtColor(frame[y0 : y0 + h, x0 : x0 + w], cv2.COLOR_BGR2GRAY)
        ink = gray <= 16
        for fill_gray in (25, 40, 55, 80, 120, 180):
            for fill_ratio in (0.0, 0.35, 0.7, 1.0):
                with self.subTest(fill_gray=fill_gray, fill_ratio=fill_ratio):
                    altered = frame.copy()
                    region = altered[y0 : y0 + h, x0 : x0 + w]
                    split = int(w * fill_ratio)
                    region[:, :split] = (fill_gray, fill_gray, fill_gray)
                    region[:, split:] = (220, 220, 220)
                    region[ink] = (0, 0, 0)
                    previous = StatusPanelValues(
                        hp=2260, hp_max=3348,
                        sp=0, sp_max=430,
                        weight=0, weight_max=2730,
                        panel_origin=origin,
                    )
                    values = read_status_panel(
                        altered, origin=origin, skip_hp=True, previous=previous
                    )
                    self.assertIsNotNone(values)
                    assert values is not None
                    self.assertEqual(values.sp, 430)
                    self.assertEqual(values.hp, 2260)


if __name__ == "__main__":
    unittest.main()
