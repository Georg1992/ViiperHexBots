"""Inventory / storage UI template matching.

Small AHK-era BMPs under ``assets/UI/*.bmp`` plus ``InventoryPanel.png``,
which anchors the Use-tab item grid (8×6, 32px pitch) measured from that crop.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from pybot.paths import ASSETS_DIR

UI_DIR = ASSETS_DIR / "UI"

# AHK ImageSearch is exact. SQDIFF_NORMED: 0 = perfect.
TEMPLATE_MATCH_MAX_SQDIFF = 0.02
# Slightly looser for the inventory title/header (anti-alias / skin variance).
PANEL_HEADER_MAX_SQDIFF = 0.05

# Legacy name used by AHK cell checks; pitch of the Use-tab icon grid.
CELL_SIZE_PX = 32

INVENTORY_PANEL_FILE = "InventoryPanel.png"
# Stable title-bar strip inside InventoryPanel.png (avoids item-dependent miss).
INV_HEADER_X = 40
INV_HEADER_Y = 0
INV_HEADER_W = 160
INV_HEADER_H = 28
# First Use-tab icon center relative to panel top-left (measured on asset).
INV_SLOT_ORIGIN_X = 56
INV_SLOT_ORIGIN_Y = 32
INV_SLOT_PITCH = 32
INV_COLS = 8
INV_ROWS = 6
INV_SLOT_HALF = 12
# Cursor tip relative to slot center: bottom-left of the icon box so the
# arrow body does not cover the item (slot still receives hover / Alt+RMB).
INV_SLOT_AIM_DX = -(INV_SLOT_HALF - 2)
INV_SLOT_AIM_DY = INV_SLOT_HALF - 2

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


@dataclass(frozen=True)
class InventoryPanelHit:
    """Top-left of the inventory window in client-frame coordinates."""

    x: int
    y: int
    width: int
    height: int

    def slot_center(self, col: int, row: int) -> tuple[int, int]:
        if not (0 <= col < INV_COLS and 0 <= row < INV_ROWS):
            raise IndexError(f"slot out of range: col={col} row={row}")
        return (
            self.x + INV_SLOT_ORIGIN_X + col * INV_SLOT_PITCH,
            self.y + INV_SLOT_ORIGIN_Y + row * INV_SLOT_PITCH,
        )

    def slot_aim(self, col: int, row: int) -> tuple[int, int]:
        """Bottom-left aim point that keeps the item icon uncovered."""
        cx, cy = self.slot_center(col, row)
        return cx + INV_SLOT_AIM_DX, cy + INV_SLOT_AIM_DY

    def iter_slot_centers(self):
        for row in range(INV_ROWS):
            for col in range(INV_COLS):
                cx, cy = self.slot_center(col, row)
                yield col, row, cx, cy


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


@lru_cache(maxsize=1)
def _load_inventory_panel() -> np.ndarray:
    path = UI_DIR / INVENTORY_PANEL_FILE
    panel = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if panel is None or panel.size == 0:
        raise FileNotFoundError(f"missing inventory panel asset: {path}")
    return panel


@lru_cache(maxsize=1)
def _load_inventory_header() -> tuple[np.ndarray, int, int]:
    """Return (header_bgr, offset_x, offset_y) within the full panel."""
    panel = _load_inventory_panel()
    h, w = panel.shape[:2]
    x0 = INV_HEADER_X
    y0 = INV_HEADER_Y
    x1 = min(w, x0 + INV_HEADER_W)
    y1 = min(h, y0 + INV_HEADER_H)
    header = panel[y0:y1, x0:x1]
    if header.size == 0:
        raise RuntimeError("inventory panel header crop is empty")
    return header, x0, y0


def clear_template_cache() -> None:
    """Drop cached inventory templates (tests / asset reloads)."""
    _load_templates.cache_clear()
    _load_inventory_panel.cache_clear()
    _load_inventory_header.cache_clear()


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


def find_inventory_panel(
    frame_bgr: np.ndarray,
    *,
    max_sqdiff: float = PANEL_HEADER_MAX_SQDIFF,
) -> InventoryPanelHit | None:
    """Locate the inventory window via its title-bar header crop."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    header, off_x, off_y = _load_inventory_header()
    panel = _load_inventory_panel()
    if (
        frame_bgr.shape[0] < header.shape[0]
        or frame_bgr.shape[1] < header.shape[1]
    ):
        return None
    result = cv2.matchTemplate(frame_bgr, header, cv2.TM_SQDIFF_NORMED)
    min_val, _max_val, min_loc, _max_loc = cv2.minMaxLoc(result)
    if min_val > max_sqdiff:
        return None
    hx, hy = int(min_loc[0]), int(min_loc[1])
    return InventoryPanelHit(
        x=hx - off_x,
        y=hy - off_y,
        width=int(panel.shape[1]),
        height=int(panel.shape[0]),
    )


def require_inventory_panel(
    frame_bgr: np.ndarray,
    *,
    max_sqdiff: float = PANEL_HEADER_MAX_SQDIFF,
) -> InventoryPanelHit:
    hit = find_inventory_panel(frame_bgr, max_sqdiff=max_sqdiff)
    if hit is None:
        raise InventoryUiError("inventory panel not found")
    return hit


def is_inventory_open(
    frame_bgr: np.ndarray,
    *,
    max_sqdiff: float = PANEL_HEADER_MAX_SQDIFF,
) -> bool:
    """True when the Inventory window title bar is visible."""
    return find_inventory_panel(frame_bgr, max_sqdiff=max_sqdiff) is not None


def is_storage_open(
    frame_bgr: np.ndarray,
    *,
    max_sqdiff: float = TEMPLATE_MATCH_MAX_SQDIFF,
) -> bool:
    """True when the storage window close button (``close_img``) is visible."""
    return find_template(frame_bgr, "close", max_sqdiff=max_sqdiff) is not None


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


def slot_contains_template(
    frame_bgr: np.ndarray,
    name: str,
    slot_cx: int,
    slot_cy: int,
    *,
    half: int = INV_SLOT_HALF,
    max_sqdiff: float = TEMPLATE_MATCH_MAX_SQDIFF,
) -> bool:
    """True if *name* is inside the icon box of one inventory slot."""
    return template_in_region(
        frame_bgr,
        name,
        slot_cx - half,
        slot_cy - half,
        slot_cx + half,
        slot_cy + half,
        max_sqdiff=max_sqdiff,
    )


def find_wing_in_slot(
    frame_bgr: np.ndarray,
    slot_cx: int,
    slot_cy: int,
    *,
    half: int = INV_SLOT_HALF,
    max_sqdiff: float = TEMPLATE_MATCH_MAX_SQDIFF,
) -> tuple[int, int] | None:
    """Return top-left of ``wing`` inside one slot icon box, or None."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    h, w = frame_bgr.shape[:2]
    x0 = max(0, slot_cx - half)
    y0 = max(0, slot_cy - half)
    x1 = min(w, slot_cx + half)
    y1 = min(h, slot_cy + half)
    if x1 <= x0 or y1 <= y0:
        return None
    loc = find_template(
        frame_bgr[y0:y1, x0:x1], "wing", max_sqdiff=max_sqdiff
    )
    if loc is None:
        return None
    return x0 + loc[0], y0 + loc[1]


def find_wings_in_use_grid(
    frame_bgr: np.ndarray,
    panel: InventoryPanelHit | None = None,
) -> list[tuple[int, int, int, int]]:
    """Return ``(col, row, aim_x, aim_y)`` for each Use-tab wing slot.

    Aim is the slot bottom-left (icon stays uncovered under the cursor).
    """
    hit = panel or find_inventory_panel(frame_bgr)
    if hit is None:
        return []
    found: list[tuple[int, int, int, int]] = []
    for col, row, cx, cy in hit.iter_slot_centers():
        if find_wing_in_slot(frame_bgr, cx, cy) is None:
            continue
        aim_x, aim_y = hit.slot_aim(col, row)
        found.append((col, row, aim_x, aim_y))
    return found


def find_storage_wing(
    frame_bgr: np.ndarray,
    panel: InventoryPanelHit | None = None,
) -> tuple[int, int] | None:
    """Locate a fly-wing icon outside the inventory panel (storage list)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    templates = _load_templates()
    tpl = templates["wing"]
    if (
        frame_bgr.shape[0] < tpl.shape[0]
        or frame_bgr.shape[1] < tpl.shape[1]
    ):
        return None
    result = cv2.matchTemplate(frame_bgr, tpl, cv2.TM_SQDIFF_NORMED)
    hit = panel or find_inventory_panel(frame_bgr)
    th, tw = tpl.shape[:2]
    # Walk best matches until one sits outside the inventory window.
    flat = result.ravel().copy()
    for _ in range(32):
        idx = int(np.argmin(flat))
        min_val = float(flat[idx])
        if min_val > TEMPLATE_MATCH_MAX_SQDIFF:
            return None
        y, x = np.unravel_index(idx, result.shape)
        loc = (int(x), int(y))
        if hit is None:
            return loc
        cx = loc[0] + tw // 2
        cy = loc[1] + th // 2
        inside = (
            hit.x <= cx < hit.x + hit.width
            and hit.y <= cy < hit.y + hit.height
        )
        if not inside:
            return loc
        # Suppress this hit and keep searching.
        y0 = max(0, y - th)
        y1 = min(result.shape[0], y + th)
        x0 = max(0, x - tw)
        x1 = min(result.shape[1], x + tw)
        result[y0:y1, x0:x1] = 1.0
        flat = result.ravel()
    return None


def slot_looks_empty(
    frame_bgr: np.ndarray,
    slot_cx: int,
    slot_cy: int,
    *,
    half: int = INV_SLOT_HALF,
) -> bool:
    """Heuristic empty check: bright, low-contrast icon box (no item glyph)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return False
    h, w = frame_bgr.shape[:2]
    x0 = max(0, slot_cx - half)
    y0 = max(0, slot_cy - half)
    x1 = min(w, slot_cx + half)
    y1 = min(h, slot_cy + half)
    if x1 <= x0 or y1 <= y0:
        return False
    crop = frame_bgr[y0:y1, x0:x1]
    mean = float(crop.mean())
    std = float(crop.std())
    return mean >= 230.0 and std <= 25.0
