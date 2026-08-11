"""Background status-panel capture/OCR helpers.

This module deliberately contains no Tk calls.  MainWindow starts one short
worker at a time and applies the returned immutable result on the UI thread.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, replace

from pybot.app.win32_util import client_rect_screen, is_window_minimized, window_exists
from pybot.recognition.capture import ui_capture_region
from pybot.recognition.ui.status_panel import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    VALUE_ROWS_ROI,
    StatusPanelValues,
    find_status_panel,
    read_status_panel,
    read_status_panel_hp,
    read_status_panel_sp,
    read_status_panel_fixed_rois,
)


@dataclass(frozen=True)
class StatusPanelReadResult:
    """Immutable result produced by one background panel read."""

    hwnd: int
    state: str
    generation: int = 0
    client_left: int = 0
    client_top: int = 0
    client_width: int = 0
    client_height: int = 0
    values: StatusPanelValues | None = None
    hp: tuple[int, int] | None = None
    sp: tuple[int, int] | None = None
    full_refresh: bool = False
    # Optional diagnostic retained after the original positional fields.
    error: str | None = None


STATUS_PANEL_READ_TIMEOUT_S = 6.0
_GEOMETRY_PROBE_TIMEOUT_S = 0.25
# Keep a phase marker so a read timeout says what the synchronous Win32/capture
# work was doing (geometry, capture, or OCR parsing) instead of reducing every
# failure to the same generic timeout.
_NATIVE_PHASE_LOCK = threading.Lock()
_NATIVE_PHASE = "idle"
_NATIVE_PHASE_STARTED = 0.0
_NATIVE_PHASE_DETAIL = ""


def _set_native_phase(
    phase: str,
    detail: str = "",
    *,
    preserve_started: bool = False,
) -> None:
    global _NATIVE_PHASE, _NATIVE_PHASE_STARTED, _NATIVE_PHASE_DETAIL
    with _NATIVE_PHASE_LOCK:
        same_phase = _NATIVE_PHASE == phase
        _NATIVE_PHASE = phase
        if not preserve_started or not same_phase:
            _NATIVE_PHASE_STARTED = time.monotonic()
        _NATIVE_PHASE_DETAIL = detail


class _GeometryRequest:
    __slots__ = ("hwnd", "done", "state", "client")

    def __init__(self, hwnd: int) -> None:
        self.hwnd = hwnd
        self.done = threading.Event()
        self.state = "client_missing"
        self.client: tuple[int, int, int, int] | None = None


class _ClientGeometryChannel:
    """Bounded Win32 geometry channel that can retire a wedged probe."""

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._queue: queue.Queue[_GeometryRequest | None] = queue.Queue(maxsize=1)
        self._worker_started = False

    @staticmethod
    def _read(request: _GeometryRequest) -> None:
        if not request.hwnd or not window_exists(request.hwnd):
            request.state = "inactive"
            return
        if is_window_minimized(request.hwnd):
            request.state = "inactive"
            return
        request.client = client_rect_screen(request.hwnd)
        request.state = "ok" if request.client is not None else "client_missing"

    def _loop(self, work_queue: queue.Queue[_GeometryRequest | None]) -> None:
        while True:
            request = work_queue.get()
            if request is None:
                return
            try:
                self._read(request)
            except BaseException:
                request.state = "client_missing"
            finally:
                request.done.set()

    def _ensure_worker(self, work_queue: queue.Queue[_GeometryRequest | None]) -> None:
        with self._state_lock:
            if self._worker_started or work_queue is not self._queue:
                return
            self._worker_started = True
        threading.Thread(
            target=self._loop,
            args=(work_queue,),
            name="ui-status-geometry",
            daemon=True,
        ).start()

    def _retire(self, work_queue: queue.Queue[_GeometryRequest | None]) -> None:
        with self._state_lock:
            if work_queue is not self._queue:
                return
            self._queue = queue.Queue(maxsize=1)
            self._worker_started = False
            try:
                work_queue.put_nowait(None)
            except queue.Full:
                pass

    def read(
        self,
        hwnd: int,
        *,
        timeout_s: float = _GEOMETRY_PROBE_TIMEOUT_S,
    ) -> tuple[str, tuple[int, int, int, int] | None]:
        with self._state_lock:
            work_queue = self._queue
        self._ensure_worker(work_queue)
        request = _GeometryRequest(hwnd)
        try:
            work_queue.put_nowait(request)
        except queue.Full:
            self._retire(work_queue)
            return "timeout", None
        if not request.done.wait(max(0.0, timeout_s)):
            self._retire(work_queue)
            return "timeout", None
        return request.state, request.client


_geometry_channel = _ClientGeometryChannel()


def read_status_panel_snapshot(
    hwnd: int,
    confirmed: StatusPanelValues | None,
    *,
    refresh_max: bool,
    timeout_s: float = STATUS_PANEL_READ_TIMEOUT_S,
    client_hint: tuple[int, int, int, int] | None = None,
    refresh_client: bool = False,
    allow_partial: bool = True,
) -> StatusPanelReadResult:
    """Capture and parse one panel snapshot without touching Tk.

    OCR reads screen pixels via mss, so it works whenever the game window is
    rendered on screen — it must NOT depend on the window being the foreground
    window. The only truly unreadable state is a minimized (off-screen) window.
    """
    # Leave a small return margin so the native helper finishes before the
    # caller-side bounded wait expires.
    deadline = time.monotonic() + max(0.0, float(timeout_s) - 0.05)
    # Geometry probes run through _geometry_channel below. Cached reads skip
    # them entirely; initial/refresh probes are retired after a short wait so
    # a wedged game window cannot strand the OCR worker.

    # Once the panel has been found, prefer the last known client rectangle.
    # This avoids calling GetClientRect/ClientToScreen on every 200 ms poll;
    # those synchronous Win32 calls are the part that can wedge after a
    # teleport/loading transition. A cached rectangle is safe because the
    # panel origin is verified against the captured pixels before publishing.
    client = client_hint
    # The client/status-panel layout is session-static. Once a confirmed
    # anchor exists, never switch the live reader back to a full-client header
    # search because of a transient teleport frame; keep sampling the same
    # value band and let the next frame publish fresh SP.
    if client is None:
        _set_native_phase(
            "geometry",
            "refresh=1" if refresh_client else "refresh=0",
        )
        state, refreshed_client = _geometry_channel.read(
            hwnd,
            timeout_s=min(
                _GEOMETRY_PROBE_TIMEOUT_S,
                max(0.0, deadline - time.monotonic()),
            ),
        )
        _set_native_phase("geometry.result", f"state={state}")
        if state == "inactive" and client is None:
            return StatusPanelReadResult(hwnd=hwnd, state="inactive")
        if refreshed_client is not None:
            client = refreshed_client
        elif client is None:
            return StatusPanelReadResult(hwnd=hwnd, state="client_missing")

    left, top, width, height = client
    if confirmed is not None:
        ox, oy = confirmed.panel_origin
        panel_in_client = (
            ox >= 0
            and oy >= 0
            and ox + PANEL_WIDTH <= width
            and oy + PANEL_HEIGHT <= height
        )
        if not panel_in_client:
            return StatusPanelReadResult(
                hwnd=hwnd,
                state="roi_missing",
                client_left=left,
                client_top=top,
                client_width=width,
                client_height=height,
            )
        value_x, value_y, value_w, value_h = VALUE_ROWS_ROI
        _set_native_phase(
            "capture.fixed_value_band",
            f"x={left + ox + value_x} y={top + oy + value_y} "
            f"w={value_w} h={value_h}",
        )
        panel_frame = ui_capture_region(
            left + ox + value_x,
            top + oy + value_y,
            value_w,
            value_h,
        )
        if panel_frame is not None and panel_frame.size > 0:
            _set_native_phase(
                "parse.fixed_rois",
                f"refresh_max={int(refresh_max)} "
                f"shape={panel_frame.shape[1]}x{panel_frame.shape[0]}",
            )
            values = read_status_panel_fixed_rois(
                panel_frame,
                origin=(-value_x, -value_y),
                previous=confirmed,
                refresh_max=refresh_max,
                deadline=deadline,
                telemetry=lambda detail: _set_native_phase(
                    "parse.fixed_rois",
                    f"refresh_max={int(refresh_max)} {detail}",
                    preserve_started=True,
                ),
            )
            if values is not None:
                return StatusPanelReadResult(
                    hwnd=hwnd,
                    state="values",
                    client_left=left,
                    client_top=top,
                    client_width=width,
                    client_height=height,
                    values=replace(values, panel_origin=(ox, oy)),
                    full_refresh=refresh_max,
                )
            if not allow_partial:
                return StatusPanelReadResult(
                    hwnd=hwnd,
                    state="roi_missing",
                    client_left=left,
                    client_top=top,
                    client_width=width,
                    client_height=height,
                )
            # Compatibility path for callers that explicitly allow partial
            # reads. The live producer disables this: it must be one simple
            # full-value parse -> storage write, never a fallback cascade.
            _set_native_phase("parse.fixed_hp_sp_fallback")
            hp = read_status_panel_hp(
                panel_frame,
                origin=(-value_x, -value_y),
                deadline=deadline,
            )
            sp = read_status_panel_sp(
                panel_frame,
                origin=(-value_x, -value_y),
                previous=confirmed,
                refresh_max=refresh_max,
                deadline=deadline,
            )
            if hp is not None or sp is not None:
                return StatusPanelReadResult(
                    hwnd=hwnd,
                    state=(
                        "hp_sp_only"
                        if hp is not None and sp is not None
                        else "sp_only"
                        if sp is not None
                        else "hp_only"
                    ),
                    client_left=left,
                    client_top=top,
                    client_width=width,
                    client_height=height,
                    hp=hp,
                    sp=sp,
                )
        # A failed fixed read is a soft ROI miss. Keep the confirmed anchor
        # and let the feed retry it; only a miss streak requests a search.
        return StatusPanelReadResult(
            hwnd=hwnd,
            state="roi_missing",
            client_left=left,
            client_top=top,
            client_width=width,
            client_height=height,
        )
    _set_native_phase(
        "capture.full_client",
        f"x={left} y={top} w={width} h={height}",
    )
    frame = ui_capture_region(left, top, width, height)
    if frame is None or frame.size == 0:
        return StatusPanelReadResult(
            hwnd=hwnd,
            state="panel_missing",
            client_left=left,
            client_top=top,
            client_width=width,
            client_height=height,
        )

    _set_native_phase(
        "parse.find_header",
        f"shape={frame.shape[1]}x{frame.shape[0]}",
    )
    origin = find_status_panel(frame, deadline=deadline)
    if origin is None:
        return StatusPanelReadResult(
            hwnd=hwnd,
            state="panel_missing",
            client_left=left,
            client_top=top,
            client_width=width,
            client_height=height,
        )

    _set_native_phase("parse.full_values", f"origin={origin[0]},{origin[1]}")
    values = read_status_panel(frame, origin=origin, deadline=deadline)
    if values is None:
        if not allow_partial:
            return StatusPanelReadResult(
                hwnd=hwnd,
                state="panel_open_digits_missing",
                client_left=left,
                client_top=top,
                client_width=width,
                client_height=height,
            )
        _set_native_phase("parse.full_hp_sp_fallback")
        hp = read_status_panel_hp(
            frame,
            origin=origin,
            deadline=deadline,
        )
        sp = read_status_panel_sp(
            frame,
            origin=origin,
            previous=confirmed,
            refresh_max=refresh_max,
            deadline=deadline,
        )
        return StatusPanelReadResult(
            hwnd=hwnd,
            state=(
                "hp_sp_only"
                if hp is not None and sp is not None
                else "sp_only"
                if sp is not None
                else "hp_only"
                if hp is not None
                else "panel_open_digits_missing"
            ),
            client_left=left,
            client_top=top,
            client_width=width,
            client_height=height,
            hp=hp,
            sp=sp,
        )

    return StatusPanelReadResult(
        hwnd=hwnd,
        state="values",
        client_left=left,
        client_top=top,
        client_width=width,
        client_height=height,
        values=values,
        full_refresh=True,
    )
