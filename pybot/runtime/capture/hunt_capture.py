"""Capture hunt ROI from a game window handle."""

from __future__ import annotations

import ctypes
import queue
import threading
import time
from ctypes import wintypes

import numpy as np

from pybot.runtime.capture.window_roi import HuntRoi, hunt_roi_from_client_rect
from pybot.config.runtime import HuntRuntimeConfig

user32 = None

# A Win32 geometry call should never be allowed to pin discovery/tracking.
# The worker is deliberately single-threaded because the APIs operate on one
# hwnd, while callers receive the last valid ROI if a native call is slow.
_GEOMETRY_WAIT_S = 0.10
_GEOMETRY_MAX_STALE_S = 1.0
_GEOMETRY_QUEUE_MAX = 1


class _GeometryRequest:
    __slots__ = ("done", "client")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.client: tuple[int, int, int, int] | None = None


def _ensure_user32():
    global user32
    if user32 is not None:
        return user32
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        raise RuntimeError("HuntWindowCapture requires Windows (ctypes.windll)")
    user32 = windll.user32
    return user32


class HuntWindowCapture:
    def __init__(self, config: HuntRuntimeConfig) -> None:
        self._config = config
        self._search_range_cells = config.search_range_cells
        self._range_lock = threading.Lock()
        self._geometry_queue: "queue.Queue[_GeometryRequest | None]" = queue.Queue(
            maxsize=_GEOMETRY_QUEUE_MAX
        )
        self._geometry_worker_started = False
        self._geometry_state_lock = threading.Lock()
        self._last_client_rect: tuple[int, int, int, int] | None = None
        self._last_client_rect_at = 0.0
        self._last_client_rect_lock = threading.Lock()

    def set_search_range_cells(self, cells: int) -> None:
        with self._range_lock:
            self._search_range_cells = cells

    @property
    def hwnd(self) -> int:
        return self._config.hwnd

    def is_valid(self) -> bool:
        """Return whether a window handle is configured.

        Do not synchronously call ``IsWindow`` here: this method is called on
        both hot observer loops. The bounded geometry worker validates the
        handle as part of the next ROI sample.
        """
        return bool(self._config.hwnd)

    def _read_client_rect_screen(self) -> tuple[int, int, int, int] | None:
        """Read geometry on the dedicated worker; never call from observers."""
        u32 = _ensure_user32()
        hwnd = self._config.hwnd
        if not hwnd or not u32.IsWindow(hwnd):
            return None

        client_rect = wintypes.RECT()
        if not u32.GetClientRect(hwnd, ctypes.byref(client_rect)):
            return None

        origin = wintypes.POINT(0, 0)
        if not u32.ClientToScreen(hwnd, ctypes.byref(origin)):
            return None

        client_w = client_rect.right - client_rect.left
        client_h = client_rect.bottom - client_rect.top
        if client_w <= 0 or client_h <= 0:
            return None
        return origin.x, origin.y, client_w, client_h

    def _geometry_worker_loop(
        self, work_queue: "queue.Queue[_GeometryRequest | None]"
    ) -> None:
        while True:
            request = work_queue.get()
            if request is None:
                return
            try:
                request.client = self._read_client_rect_screen()
            except BaseException:
                # A failed native geometry call is an unavailable sample, not
                # a reason to kill the observer threads.
                request.client = None
            finally:
                request.done.set()

    def _ensure_geometry_worker(
        self, work_queue: "queue.Queue[_GeometryRequest | None]"
    ) -> None:
        with self._geometry_state_lock:
            if self._geometry_worker_started or work_queue is not self._geometry_queue:
                return
            self._geometry_worker_started = True
        threading.Thread(
            target=self._geometry_worker_loop,
            args=(work_queue,),
            name="hunt-window-geometry",
            daemon=True,
        ).start()

    def _retire_geometry_worker(
        self, work_queue: "queue.Queue[_GeometryRequest | None]"
    ) -> None:
        """Rotate a geometry worker whose native Win32 call did not return."""
        with self._geometry_state_lock:
            if work_queue is not self._geometry_queue:
                return
            self._geometry_queue = queue.Queue(maxsize=_GEOMETRY_QUEUE_MAX)
            self._geometry_worker_started = False
            try:
                work_queue.put_nowait(None)
            except queue.Full:
                pass

    def get_client_rect_screen(self) -> tuple[int, int, int, int] | None:
        """Return fresh geometry without blocking the observation workers.

        A slow or wedged ``GetClientRect``/``ClientToScreen`` remains isolated
        on the geometry daemon. The caller waits only briefly and falls back to
        the last valid rectangle; if there is no cache yet it returns ``None``.
        """
        with self._geometry_state_lock:
            work_queue = self._geometry_queue
        self._ensure_geometry_worker(work_queue)
        request = _GeometryRequest()
        queued = True
        try:
            work_queue.put_nowait(request)
        except queue.Full:
            queued = False
        if queued and request.done.wait(_GEOMETRY_WAIT_S):
            client = request.client
            if client is not None:
                with self._last_client_rect_lock:
                    self._last_client_rect = client
                    self._last_client_rect_at = time.monotonic()
                return client
            return None
        if queued:
            self._retire_geometry_worker(work_queue)
        with self._last_client_rect_lock:
            if (
                self._last_client_rect is not None
                and time.monotonic() - self._last_client_rect_at
                <= _GEOMETRY_MAX_STALE_S
            ):
                return self._last_client_rect
        return None

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

    def capture_roi(
        self,
        roi: HuntRoi,
        *,
        observer: str = "runtime",
    ) -> np.ndarray | None:
        """Capture one ROI on the caller's observer channel.

        Discovery and coordinate tracking pass different channel names so a
        blocked native grab in one observer cannot hold the other observer's
        frame hostage. ``runtime`` retains the legacy shared channel for
        callers outside those observers.
        """
        if observer == "runtime":
            from pybot.recognition.capture import capture_region

            return capture_region(roi.x, roi.y, roi.w, roi.h)
        from pybot.recognition.capture import observation_capture_region

        return observation_capture_region(
            roi.x, roi.y, roi.w, roi.h, observer=observer
        )

    def capture_client(self) -> np.ndarray | None:
        """Capture the full game client (e.g. Basic Info status panel OCR)."""
        from pybot.recognition.capture import capture_region

        client = self.get_client_rect_screen()
        if client is None:
            return None
        left, top, width, height = client
        return capture_region(left, top, width, height)
