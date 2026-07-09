"""Capture hunt ROI from a game window handle."""

from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes

import numpy as np

from pybot.runtime.capture.window_roi import HuntRoi, hunt_roi_from_client_rect
from pybot.runtime.config import HuntRuntimeConfig

user32 = ctypes.windll.user32


class HuntWindowCapture:
    def __init__(self, config: HuntRuntimeConfig) -> None:
        self._config = config
        self._search_range_cells = config.search_range_cells
        self._range_lock = threading.Lock()

    def set_search_range_cells(self, cells: int) -> None:
        with self._range_lock:
            self._search_range_cells = cells

    @property
    def hwnd(self) -> int:
        return self._config.hwnd

    def is_valid(self) -> bool:
        return bool(self._config.hwnd) and bool(user32.IsWindow(self._config.hwnd))

    def get_client_rect_screen(self) -> tuple[int, int, int, int] | None:
        hwnd = self._config.hwnd
        if not hwnd or not user32.IsWindow(hwnd):
            return None

        client_rect = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(client_rect)):
            return None

        origin = wintypes.POINT(0, 0)
        if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            return None

        client_w = client_rect.right - client_rect.left
        client_h = client_rect.bottom - client_rect.top
        if client_w <= 0 or client_h <= 0:
            return None
        return origin.x, origin.y, client_w, client_h

    def get_hunt_roi(self) -> HuntRoi | None:
        client = self.get_client_rect_screen()
        if client is None:
            return None
        client_left, client_top, client_w, client_h = client
        with self._range_lock:
            search_range_cells = self._search_range_cells
        return hunt_roi_from_client_rect(
            client_left,
            client_top,
            client_w,
            client_h,
            search_range_cells=search_range_cells,
            cell_size_px=self._config.cell_size_px,
        )

    def capture_roi(self, roi: HuntRoi) -> np.ndarray | None:
        from pybot.recognition.capture import capture_region

        return capture_region(roi.x, roi.y, roi.w, roi.h)
