"""Inventory / storage UI template matching (AHK HexBots image assets).

Templates live under ``assets/UI/*.bmp`` with the same filenames as the AHK bot.
Matching is intentionally strict (near-exact) to mirror AHK ``ImageSearch``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from pybot.paths import ASSETS_DIR

UI_DIR = ASSETS_DIR / "UI"

# AHK ImageSearch is exact. SQDIFF_NORMED: 0 = perfect (works for flat templates
# like empty_cell.bmp; CCOEFF_NORMED is unstable on constant patches).
TEMPLATE_MATCH_MAX_SQDIFF = 0.02
CELL_SIZE_PX = 40

TEMPLATE_FILES: dict[str, str] = {
    "use": "use_img.bmp",
    "eqp": "eqp_img.bmp",
    "etc": "etc_img.bmp",
    "close": "close_img.bmp",
    "cell1": "cell1_img.bmp",
    "wing": "wing_img.bmp",
    "ok": "ok_img.bmp",
    "empty_cell": "empty_cell_img.bmp",
}


class InventoryUiError(RuntimeError):
    """Raised when a required inventory/storage UI template is not found."""


@lru_cache(maxsize=1)
def _load_templates() -> dict[str, np.ndarray]:
    loaded: dict[str, np.ndarray] = {}
    for name, filename in TEMPLATE_FILES.items():
        path = UI_DIR / filename
        tpl = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if tpl is None or tpl.size == 0:
            raise FileNotFoundError(f"missing inventory UI template: {path}")
        loaded[name] = tpl
    return loaded


def clear_template_cache() -> None:
    """Drop cached inventory templates (tests / asset reloads)."""
    _load_templates.cache_clear()


def template_path(name: str) -> Path:
    if name not in TEMPLATE_FILES:
        raise KeyError(f"unknown inventory template: {name}")
    return UI_DIR / TEMPLATE_FILES[name]


def find_template(
    frame_bgr: np.ndarray,
    name: str,
    *,
    max_sqdiff: float = TEMPLATE_MATCH_MAX_SQDIFF,
) -> tuple[int, int] | None:
    """Return top-left of *name* in *frame_bgr*, or None if not found."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    templates = _load_templates()
    tpl = templates[name]
    if (
        frame_bgr.shape[0] < tpl.shape[0]
        or frame_bgr.shape[1] < tpl.shape[1]
    ):
        return None
    result = cv2.matchTemplate(frame_bgr, tpl, cv2.TM_SQDIFF_NORMED)
    min_val, _max_val, min_loc, _max_loc = cv2.minMaxLoc(result)
    if min_val > max_sqdiff:
        return None
    return int(min_loc[0]), int(min_loc[1])


def require_template(
    frame_bgr: np.ndarray,
    name: str,
    *,
    max_sqdiff: float = TEMPLATE_MATCH_MAX_SQDIFF,
) -> tuple[int, int]:
    """Return top-left of *name*, or raise ``InventoryUiError``."""
    loc = find_template(frame_bgr, name, max_sqdiff=max_sqdiff)
    if loc is None:
        raise InventoryUiError(f"inventory UI template not found: {name}")
    return loc


def template_in_region(
    frame_bgr: np.ndarray,
    name: str,
    left: int,
    top: int,
    right: int,
    bottom: int,
    *,
    max_sqdiff: float = TEMPLATE_MATCH_MAX_SQDIFF,
) -> bool:
    """True if *name* matches inside the inclusive-exclusive ROI."""
    if frame_bgr is None or frame_bgr.size == 0:
        return False
    h, w = frame_bgr.shape[:2]
    x0 = max(0, min(w, left))
    y0 = max(0, min(h, top))
    x1 = max(0, min(w, right))
    y1 = max(0, min(h, bottom))
    if x1 <= x0 or y1 <= y0:
        return False
    crop = frame_bgr[y0:y1, x0:x1]
    return find_template(crop, name, max_sqdiff=max_sqdiff) is not None


def cell_contains_template(
    frame_bgr: np.ndarray,
    name: str,
    cursor_x: int,
    cursor_y: int,
    *,
    cell_size: int = CELL_SIZE_PX,
    max_sqdiff: float = TEMPLATE_MATCH_MAX_SQDIFF,
) -> bool:
    """AHK ``CheckInventoryCell``: search ±cell_size/2 around the cursor."""
    half = cell_size // 2
    return template_in_region(
        frame_bgr,
        name,
        cursor_x - half,
        cursor_y - half,
        cursor_x + half,
        cursor_y + half,
        max_sqdiff=max_sqdiff,
    )
