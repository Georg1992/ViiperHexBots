"""Hunt search region"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HuntRoi:
    x: int
    y: int
    w: int
    h: int

    @property
    def center_x(self) -> int:
        return self.x + (self.w // 2)

    @property
    def center_y(self) -> int:
        return self.y + (self.h // 2)


def search_box_size_px(search_range_cells: int, cell_size_px: int) -> int:
    return search_range_cells * cell_size_px


def hunt_roi_from_client_rect(
    client_left: int,
    client_top: int,
    client_w: int,
    client_h: int,
    *,
    search_range_cells: int,
    cell_size_px: int,
) -> HuntRoi | None:
    search_size = search_box_size_px(search_range_cells, cell_size_px)
    if search_size <= 0 or client_w <= 0 or client_h <= 0:
        return None

    w = search_size
    h = search_size
    x = client_left + (client_w // 2) - (w // 2)
    y = client_top + (client_h // 2) - (h // 2)

    if x < client_left:
        x = client_left
    if y < client_top:
        y = client_top
    if x + w > client_left + client_w:
        x = client_left + client_w - w
    if y + h > client_top + client_h:
        y = client_top + client_h - h

    if w <= 0 or h <= 0:
        return None
    return HuntRoi(x=x, y=y, w=w, h=h)


def hunt_roi_from_frame_shape(
    frame_h: int,
    frame_w: int,
    *,
    search_range_cells: int,
    cell_size_px: int,
) -> HuntRoi | None:
    """Local-frame hunt ROI using the same math as live capture."""
    return hunt_roi_from_client_rect(
        0,
        0,
        frame_w,
        frame_h,
        search_range_cells=search_range_cells,
        cell_size_px=cell_size_px,
    )


def crop_frame_to_hunt_search_roi(
    frame: np.ndarray,
    *,
    search_range_cells: int,
    cell_size_px: int,
) -> np.ndarray:
    """Crop a BGR frame to the hunt search box (mirrors production capture)."""
    h, w = frame.shape[:2]
    roi = hunt_roi_from_frame_shape(
        h,
        w,
        search_range_cells=search_range_cells,
        cell_size_px=cell_size_px,
    )
    if roi is None:
        return frame
    x0 = max(0, roi.x)
    y0 = max(0, roi.y)
    x1 = min(w, roi.x + roi.w)
    y1 = min(h, roi.y + roi.h)
    if x1 <= x0 or y1 <= y0:
        return frame
    return frame[y0:y1, x0:x1].copy()


def player_ignore_box(
    roi: HuntRoi,
    cell_size_px: int,
) -> tuple[int, int, int, int]:
    """Mirrors GetMobSearchPlayerIgnore — center 2x2 cells."""
    ignore_w = cell_size_px * 2
    ignore_h = cell_size_px * 2
    ignore_x = roi.x + (roi.w // 2) - (ignore_w // 2)
    ignore_y = roi.y + (roi.h // 2) - (ignore_h // 2)
    return ignore_x, ignore_y, ignore_w, ignore_h


def point_inside_ignore(
    x: int,
    y: int,
    ignore_x: int,
    ignore_y: int,
    ignore_w: int,
    ignore_h: int,
) -> bool:
    if ignore_w <= 0 or ignore_h <= 0:
        return False
    return ignore_x <= x <= ignore_x + ignore_w and ignore_y <= y <= ignore_y + ignore_h
