"""Locate the Basic Info status panel and parse SP / Weight.

Uses OpenCV template matching for the panel header, fixed relative ROIs for
value bands, and RO digit-glyph templates under ``assets/UI/digits/``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from pybot.paths import ASSETS_DIR

UI_DIR = ASSETS_DIR / "UI"
HEADER_TEMPLATE_PATH = UI_DIR / "status_panel_header.png"
DIGITS_DIR = UI_DIR / "digits"

# Header template is cropped from StatusPanel.png at (HEADER_OFFSET_X, 0).
HEADER_OFFSET_X = 5
HEADER_MATCH_THRESHOLD = 0.85
DIGIT_MATCH_THRESHOLD = 0.90
BINARIZE_THRESHOLD = 70
# Single glyphs are typically 3–6px wide; wider blobs are touching digits.
MAX_GLYPH_WIDTH = 7

# Full Basic Info panel size (for overlay placement).
PANEL_WIDTH = 219
PANEL_HEIGHT = 143

# Value ROIs relative to the full Basic Info panel origin (x, y, w, h).
SP_ROI = (50, 68, 110, 12)
WEIGHT_ROI = (85, 118, 58, 11)


@dataclass(frozen=True)
class StatusPanelValues:
    sp: int
    sp_max: int
    weight: int
    weight_max: int
    panel_origin: tuple[int, int]


@lru_cache(maxsize=1)
def _load_header_template() -> np.ndarray:
    tpl = cv2.imread(str(HEADER_TEMPLATE_PATH), cv2.IMREAD_COLOR)
    if tpl is None or tpl.size == 0:
        raise FileNotFoundError(f"missing status panel header: {HEADER_TEMPLATE_PATH}")
    return tpl


@lru_cache(maxsize=1)
def _load_digit_templates() -> dict[str, tuple[np.ndarray, ...]]:
    if not DIGITS_DIR.is_dir():
        raise FileNotFoundError(f"missing digit templates dir: {DIGITS_DIR}")
    by_char: dict[str, list[np.ndarray]] = defaultdict(list)
    for path in sorted(DIGITS_DIR.glob("*.png")):
        stem = path.stem
        ch = "/" if stem.startswith("slash") else stem[0]
        glyph = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if glyph is None or glyph.size == 0:
            raise FileNotFoundError(f"unreadable digit template: {path}")
        by_char[ch].append(glyph)
    required = {str(d) for d in range(10)} | {"/"}
    missing = required - set(by_char)
    if missing:
        raise FileNotFoundError(f"digit templates missing chars: {sorted(missing)}")
    return {ch: tuple(glyphs) for ch, glyphs in by_char.items()}


def find_status_panel(frame_bgr: np.ndarray) -> tuple[int, int] | None:
    """Return top-left of the Basic Info panel in *frame_bgr*, or None."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    header = _load_header_template()
    if (
        frame_bgr.shape[0] < header.shape[0]
        or frame_bgr.shape[1] < header.shape[1]
    ):
        return None
    result = cv2.matchTemplate(frame_bgr, header, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val < HEADER_MATCH_THRESHOLD:
        return None
    hx, hy = int(max_loc[0]), int(max_loc[1])
    return hx - HEADER_OFFSET_X, hy


def read_status_panel(frame_bgr: np.ndarray) -> StatusPanelValues | None:
    """Locate Basic Info and parse SP/Weight ``current / max`` values."""
    origin = find_status_panel(frame_bgr)
    if origin is None:
        return None
    sp = _parse_pair(frame_bgr, origin, SP_ROI, min_width=2)
    weight = _parse_pair(frame_bgr, origin, WEIGHT_ROI, min_width=3)
    if sp is None or weight is None:
        return None
    return StatusPanelValues(
        sp=sp[0],
        sp_max=sp[1],
        weight=weight[0],
        weight_max=weight[1],
        panel_origin=origin,
    )


def _parse_pair(
    frame_bgr: np.ndarray,
    origin: tuple[int, int],
    roi: tuple[int, int, int, int],
    *,
    min_width: int,
) -> tuple[int, int] | None:
    crop = _crop_roi(frame_bgr, origin, roi)
    if crop is None:
        return None
    text = _read_digits(crop, min_width=min_width)
    if text is None or text.count("/") != 1:
        return None
    left, right = text.split("/", 1)
    if not left.isdigit() or not right.isdigit():
        return None
    return int(left), int(right)


def _crop_roi(
    frame_bgr: np.ndarray,
    origin: tuple[int, int],
    roi: tuple[int, int, int, int],
) -> np.ndarray | None:
    ox, oy = origin
    x, y, w, h = roi
    x0, y0 = ox + x, oy + y
    x1, y1 = x0 + w, y0 + h
    if x0 < 0 or y0 < 0 or y1 > frame_bgr.shape[0] or x1 > frame_bgr.shape[1]:
        return None
    return frame_bgr[y0:y1, x0:x1]


def _to_ink_mask(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _unused, mask = cv2.threshold(
        gray, BINARIZE_THRESHOLD, 255, cv2.THRESH_BINARY_INV
    )
    return mask


def _trim_empty(glyph: np.ndarray) -> np.ndarray | None:
    cols = np.where(glyph.any(axis=0))[0]
    rows = np.where(glyph.any(axis=1))[0]
    if cols.size == 0 or rows.size == 0:
        return None
    return glyph[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy()


def _split_wide_glyph(glyph: np.ndarray) -> list[np.ndarray]:
    """Split touching digits on vertical ink valleys (e.g. SP ``43`` blobs)."""
    h, w = glyph.shape
    if w <= MAX_GLYPH_WIDTH:
        trimmed = _trim_empty(glyph)
        return [trimmed] if trimmed is not None else []
    col_ink = (glyph > 0).sum(axis=0)
    cuts: list[int] = []
    x = 1
    while x < w - 1:
        if col_ink[x] <= 1 and col_ink[x] <= col_ink[x - 1] and col_ink[x] <= col_ink[x + 1]:
            cuts.append(x)
            x += 2
            continue
        x += 1
    if not cuts:
        # Two touching digits with no zero valley: cut near the midpoint.
        cuts = [w // 2]
    bounds = [0, *cuts, w]
    parts: list[np.ndarray] = []
    for left, right in zip(bounds, bounds[1:]):
        if right - left < 2:
            continue
        trimmed = _trim_empty(glyph[:, left:right])
        if trimmed is None:
            continue
        parts.extend(_split_wide_glyph(trimmed))
    return parts


def _glyph_components(mask: np.ndarray, *, min_width: int) -> list[np.ndarray]:
    # Bar chrome on the first/last rows can bridge digits; drop it.
    cleaned = mask.copy()
    cleaned[0, :] = 0
    if cleaned.shape[0] > 1:
        cleaned[-1, :] = 0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    comps: list[tuple[int, np.ndarray]] = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 3 or h < 5 or w < min_width:
            continue
        blob = cleaned[y : y + h, x : x + w].copy()
        parts = _split_wide_glyph(blob) if w > MAX_GLYPH_WIDTH else [blob]
        for part in parts:
            if part.shape[0] < 5 or part.shape[1] < min_width:
                continue
            comps.append((int(x), part))
    comps.sort(key=lambda item: item[0])
    return [glyph for _x, glyph in comps]


def _classify_glyph(glyph: np.ndarray) -> tuple[str | None, float]:
    templates = _load_digit_templates()
    best_ch: str | None = None
    best_score = -1.0
    for ch, variants in templates.items():
        for tpl in variants:
            pad_h = glyph.shape[0] + tpl.shape[0] + 4
            pad_w = glyph.shape[1] + tpl.shape[1] + 4
            pad = np.zeros((pad_h, pad_w), dtype=np.uint8)
            pad[2 : 2 + glyph.shape[0], 2 : 2 + glyph.shape[1]] = glyph
            score = float(
                cv2.minMaxLoc(
                    cv2.matchTemplate(pad, tpl, cv2.TM_CCOEFF_NORMED)
                )[1]
            )
            if score > best_score:
                best_score = score
                best_ch = ch
    return best_ch, best_score


def _read_digits(bgr: np.ndarray, *, min_width: int) -> str | None:
    mask = _to_ink_mask(bgr)
    chars: list[str] = []
    for glyph in _glyph_components(mask, min_width=min_width):
        ch, score = _classify_glyph(glyph)
        # Fail closed: skipping a weak glyph silently drops digits (e.g. 430 → 40).
        if ch is None or score < DIGIT_MATCH_THRESHOLD:
            return None
        chars.append(ch)
    if not chars:
        return None
    return "".join(chars)


def clear_template_cache() -> None:
    """Drop cached header/digit templates (tests / asset reloads)."""
    _load_header_template.cache_clear()
    _load_digit_templates.cache_clear()
