"""Locate the Basic Info status panel and parse HP / SP / Weight.

Uses OpenCV template matching for the panel header, fixed relative ROIs for
value bands, and RO digit-glyph templates under ``assets/UI/digits/``.

Two polling cadences share the same function ``read_status_panel``:

* **Full read** (no flags) — parses HP + SP + Weight current+max.
  Called every ``STATUS_PANEL_MAX_REFRESH_S`` (1 s) to refresh all maxima.

* **Fast poll** (``skip_hp=True, previous=<last full>``) — reuses HP/HP_max
  and max values from the previous full read, only OCRs current SP/Weight.
  Called every ``STATUS_PANEL_VALUE_MS`` (200 ms).  Avoiding the HP-band
  OCR on every tick prevents Gepard-sensitivity artifacts (bar fill at low
  HP being misread as digit "1").

Callers should treat ``find_status_panel`` as the source of truth for whether
Basic Info is open.
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
DIGIT_MATCH_THRESHOLD = 0.85
# RO digit cores are pure black (gray 0). SP bar fill/empty are mid-bright and
# change with SP%% — keep this well below the darkest real bar pixels (~47+).
BINARIZE_THRESHOLD = 45
# Single glyphs are typically 3–6px wide; wider blobs are touching digits.
MAX_GLYPH_WIDTH = 7
# Real digit boxes sit ~1px apart; label leftovers (e.g. trailing ``t`` of
# ``Weight``) sit farther left with a wide gap before the value digits.
MAX_LEADING_ORPHAN_GAP_PX = 6

# Full Basic Info panel size (for overlay placement).
PANEL_WIDTH = 219
PANEL_HEIGHT = 143

# Value ROIs relative to the full Basic Info panel origin (x, y, w, h).
# Padded so ±2px header/origin jitter still keeps full glyph height/width.
# HP sits directly above SP on the same digit column.
HP_ROI = (50, 45, 110, 18)
SP_ROI = (50, 66, 110, 16)
# Weight digits shift left when Zeny is long (e.g. "1,234,567z") — the first
# digit of a 5-digit weight can start as early as x≈76.  x=75 captures that
# without reaching the colon (x≈68-70) which would be misread as "/".
# Width 84 keeps the same right edge (75+84=159) as the previous (85+74=159)
# so trailing Zeny crumbs still stay below threshold and strip.
WEIGHT_ROI = (75, 116, 84, 14)

# ── Dynamic scan zones ─────────────────────────────────────────────
# These are wide enough to *always* contain the "/" separator regardless
# of digit length or Zeny shift.  Once "/" is located a tighter ROI is
# built around it with generous horizontal expansion so no digit is cropped.
HP_SCAN_ZONE   = (42, 45, 138, 18)   # x=42..180
SP_SCAN_ZONE   = (42, 66, 138, 16)   # x=42..180
WEIGHT_SCAN_ZONE = (70, 116, 110, 14) # x=70..180

# Pixels to expand left/right from "/" center — enough for 5 digits + gaps
_SLASH_LEFT_EXPAND = 52
_SLASH_RIGHT_EXPAND = 52


@dataclass(frozen=True)
class StatusPanelValues:
    hp: int
    hp_max: int
    sp: int
    sp_max: int
    weight: int | None
    weight_max: int | None
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


def read_status_panel(
    frame_bgr: np.ndarray,
    *,
    origin: tuple[int, int] | None = None,
    skip_hp: bool = False,
    previous: StatusPanelValues | None = None,
) -> StatusPanelValues | None:
    """Parse HP/SP/Weight from the Basic Info panel.

    Uses dynamic ROI anchoring on the ``/`` separator so digits are never
    cropped regardless of horizontal shift (Zeny length, digit count, etc.).

    Parameters
    ----------
    frame_bgr : np.ndarray
        Full client-area frame in BGR.
    origin : tuple[int, int] | None
        Known panel origin (skip header search).
    skip_hp : bool
        When *True*, skip the HP-band OCR and reuse HP/HP max / SP max /
        Weight max from *previous*.  ``previous`` must not be None.
    previous : StatusPanelValues | None
        Last full-read result providing max values for the fast poll.
        Required when ``skip_hp=True``.
    """
    if origin is None:
        origin = find_status_panel(frame_bgr)
    if origin is None:
        return None

    if skip_hp:
        if previous is None:
            raise ValueError("previous must be provided when skip_hp=True")
        sp = _parse_anchored(frame_bgr, origin, SP_SCAN_ZONE, min_width=2, stop_at_slash=True)
        if sp is None:
            return None
        weight = _parse_anchored(frame_bgr, origin, WEIGHT_SCAN_ZONE, min_width=3, stop_at_slash=True)
        return StatusPanelValues(
            hp=previous.hp,
            hp_max=previous.hp_max,
            sp=sp,
            sp_max=previous.sp_max,
            weight=weight,
            weight_max=previous.weight_max if weight is not None else None,
            panel_origin=origin,
        )

    # Full read — parse current+max for all three bands.
    hp = _parse_anchored(frame_bgr, origin, HP_SCAN_ZONE, min_width=2, stop_at_slash=False)
    if hp is None:
        return None
    sp = _parse_anchored(frame_bgr, origin, SP_SCAN_ZONE, min_width=2, stop_at_slash=False)
    if sp is None:
        return None
    weight = _parse_anchored(frame_bgr, origin, WEIGHT_SCAN_ZONE, min_width=3, stop_at_slash=False)
    return StatusPanelValues(
        hp=hp[0],
        hp_max=hp[1],
        sp=sp[0],
        sp_max=sp[1],
        weight=None if weight is None else weight[0],
        weight_max=None if weight is None else weight[1],
        panel_origin=origin,
    )


# ── /-anchored dynamic parsing (single unified path, no fallbacks) ──

def _find_slash_x_in_scan(
    frame_bgr: np.ndarray,
    origin: tuple[int, int],
    scan_roi: tuple[int, int, int, int],
    min_width: int,
) -> int | None:
    """Find the x-offset of ``/`` within *scan_roi* (relative to crop origin).

    Returns the **rightmost** ``/`` match to avoid label-colon artifacts
    (e.g. the colon of ``Weight:`` can be misclassified as ``/`` at x≈0).
    """
    crop = _crop_roi(frame_bgr, origin, scan_roi)
    if crop is None:
        return None
    mask = _to_ink_mask(crop)
    cleaned = _strip_bar_chrome(mask)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    rightmost: int | None = None
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 3 or h < 5 or w < min_width:
            continue
        blob = cleaned[y : y + h, x : x + w].copy()
        parts = _split_wide_glyph(blob) if w > MAX_GLYPH_WIDTH else [blob]
        for part in parts:
            trimmed = _trim_empty(part)
            if trimmed is None or trimmed.shape[0] < 5 or trimmed.shape[1] < min_width:
                continue
            ch, score = _classify_glyph(trimmed)
            if ch == "/" and score >= DIGIT_MATCH_THRESHOLD:
                cx = x + trimmed.shape[1] // 2
                if rightmost is None or cx > rightmost:
                    rightmost = cx
    return rightmost


def _build_dynamic_roi(
    slash_x: int,
    scan_roi: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Build a tighter ROI around ``/`` with generous horizontal expansion.

    Returns ``(x, y, w, h)`` in origin-relative coordinates.
    """
    sx, sy, sw, sh = scan_roi
    left = max(0, slash_x - _SLASH_LEFT_EXPAND)
    right = min(sw, slash_x + _SLASH_RIGHT_EXPAND)
    return (sx + left, sy, right - left, sh)


def _parse_anchored(
    frame_bgr: np.ndarray,
    origin: tuple[int, int],
    scan_roi: tuple[int, int, int, int],
    *,
    min_width: int,
    stop_at_slash: bool,
) -> int | tuple[int, int] | None:
    """Parse OCR values using ``/``-anchored dynamic ROI.

    Finds the ``/`` separator inside *scan_roi*, builds a bounded ROI
    around it with ``_SLASH_LEFT_EXPAND`` / ``_SLASH_RIGHT_EXPAND``, then
    reads digits.

    When *stop_at_slash* is ``True`` returns only the current value (``int``).
    Otherwise returns ``(current, max)`` pair.

    Returns ``None`` when ``/`` is not found or any digit validation fails.
    """
    slash_x = _find_slash_x_in_scan(frame_bgr, origin, scan_roi, min_width)
    if slash_x is None:
        return None

    crop = _crop_roi(frame_bgr, origin, _build_dynamic_roi(slash_x, scan_roi))
    if crop is None:
        return None

    text = _read_digits(crop, min_width=min_width, stop_at_slash=stop_at_slash)
    if text is None:
        return None

    if stop_at_slash:
        if not text.isdigit():
            return None
        return int(text)

    if text.count("/") != 1:
        return None
    left, right = text.split("/", 1)
    if not left.isdigit() or not right.isdigit():
        return None
    current, maximum = int(left), int(right)
    if not _valid_pair(current, maximum):
        return None
    return current, maximum


def _valid_pair(current: int, maximum: int) -> bool:
    """Reject absurd OCR (same checks as Belarus statusui validateValues)."""
    return maximum > 0 and 0 <= current <= maximum


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
    """Digit ink mask that ignores SP/Weight bar fill colors.

    Near-black cores use local contrast vs a morphologically closed
    background (fill/empty change with SP%% without becoming ink).
    Overweight weight text is saturated red on the same light panel.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 9))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    darker = cv2.subtract(background, gray)
    near_black = gray <= BINARIZE_THRESHOLD
    strong_contrast = darker >= 28
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[near_black & strong_contrast] = 255
    # Pure black cores always count even if closing underestimates contrast.
    mask[gray <= 16] = 255
    # Overweight Weight digits: saturated red (e.g. BGR 0,0,240), not near-black.
    blue, green, red = cv2.split(bgr)
    red_ink = (red >= 180) & (green <= 80) & (blue <= 80)
    mask[red_ink] = 255
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


def _strip_bar_chrome(mask: np.ndarray) -> np.ndarray:
    """Clear horizontal bar-border rows that would bridge adjacent glyphs."""
    cleaned = mask.copy()
    h, w = cleaned.shape
    if h == 0 or w == 0:
        return cleaned
    min_chrome = max(3, int(w * 0.65))
    row_ink = (cleaned > 0).sum(axis=1)
    cleaned[row_ink >= min_chrome, :] = 0
    return cleaned


def _drop_leading_orphan_glyphs(
    comps: list[tuple[int, np.ndarray]],
) -> list[tuple[int, np.ndarray]]:
    """Drop left-side label fragments separated from the digit cluster by a gap.

    Label crumbs can pack tightly (colon / ``t`` fragments) and sit before a
    large gap into the real digits. Cut at large gaps that appear *before* the
    last current-value glyph — never at the current-to-``/`` spacing (7–12px).
    """
    if len(comps) < 2:
        return comps
    slash_i: int | None = None
    for index, (_x, glyph) in enumerate(comps):
        ch, score = _classify_glyph(glyph)
        if ch == "/" and score >= DIGIT_MATCH_THRESHOLD:
            slash_i = index
            break
    if slash_i is None:
        while len(comps) >= 2:
            x0, glyph0 = comps[0]
            x1, _glyph1 = comps[1]
            gap = x1 - (x0 + glyph0.shape[1])
            if gap <= MAX_LEADING_ORPHAN_GAP_PX:
                break
            comps = comps[1:]
        return comps
    cut = 0
    for index in range(max(0, slash_i - 1)):
        x0, glyph0 = comps[index]
        x1, _glyph1 = comps[index + 1]
        gap = x1 - (x0 + glyph0.shape[1])
        if gap > MAX_LEADING_ORPHAN_GAP_PX:
            cut = index + 1
    return comps[cut:]


def _glyph_components(mask: np.ndarray, *, min_width: int) -> list[np.ndarray]:
    cleaned = _strip_bar_chrome(mask)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    comps: list[tuple[int, np.ndarray]] = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 3 or h < 5 or w < min_width:
            continue
        blob = cleaned[y : y + h, x : x + w].copy()
        if w > MAX_GLYPH_WIDTH:
            parts = _split_wide_glyph(blob)
        else:
            trimmed = _trim_empty(blob)
            parts = [trimmed] if trimmed is not None else []
        for part in parts:
            if part.shape[0] < 5 or part.shape[1] < min_width:
                continue
            comps.append((int(x), part))
    comps.sort(key=lambda item: item[0])
    comps = _drop_leading_orphan_glyphs(comps)
    return [glyph for _x, glyph in comps]


def _classify_glyph(glyph: np.ndarray) -> tuple[str | None, float]:
    templates = _load_digit_templates()
    best_ch: str | None = None
    best_score = -1.0
    gh, gw = glyph.shape[:2]
    for ch, variants in templates.items():
        for tpl in variants:
            th, tw = tpl.shape[:2]
            # Belarus statusui: skip wildly different scales (dot vs digit).
            if gw * 2 < tw or tw * 2 < gw or gh * 2 < th or th * 2 < gh:
                continue
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


def _read_digits(
    bgr: np.ndarray,
    *,
    min_width: int,
    stop_at_slash: bool,
) -> str | None:
    mask = _to_ink_mask(bgr)
    classified: list[tuple[str | None, float]] = [
        _classify_glyph(glyph)
        for glyph in _glyph_components(mask, min_width=min_width)
    ]
    # Drop trailing edge chrome that fails the digit threshold (wider ROI / jitter).
    while classified and (
        classified[-1][0] is None or classified[-1][1] < DIGIT_MATCH_THRESHOLD
    ):
        classified.pop()
    chars: list[str] = []
    for ch, score in classified:
        if ch is None or score < DIGIT_MATCH_THRESHOLD:
            return None
        if ch == "/":
            if stop_at_slash:
                break
            chars.append(ch)
            continue
        chars.append(ch)
    if not chars:
        return None
    return "".join(chars)


def clear_template_cache() -> None:
    """Drop cached header/digit templates (tests / asset reloads)."""
    _load_header_template.cache_clear()
    _load_digit_templates.cache_clear()
