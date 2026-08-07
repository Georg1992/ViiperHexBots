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
    verify_status_panel_at,
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
# At most one native status read may outlive its caller-side timeout. Without
# this guard, a permanently blocked Win32/capture call would leave one daemon
# helper behind per 200 ms poll after the first timeout.
_NATIVE_READ_IN_FLIGHT = threading.Lock()


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
    reanchor: bool = False,
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
    if client is None or refresh_client:
        state, refreshed_client = _geometry_channel.read(
            hwnd,
            timeout_s=min(
                _GEOMETRY_PROBE_TIMEOUT_S,
                max(0.0, deadline - time.monotonic()),
            ),
        )
        if state == "inactive" and client is None:
            return StatusPanelReadResult(hwnd=hwnd, state="inactive")
        if refreshed_client is not None:
            client = refreshed_client
        elif client is None:
            return StatusPanelReadResult(hwnd=hwnd, state="client_missing")

    left, top, width, height = client
    if confirmed is not None and not reanchor:
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
        panel_frame = ui_capture_region(
            left + ox + value_x,
            top + oy + value_y,
            value_w,
            value_h,
        )
        if panel_frame is not None and panel_frame.size > 0:
            values = read_status_panel_fixed_rois(
                panel_frame,
                origin=(-value_x, -value_y),
                previous=confirmed,
                refresh_max=refresh_max,
                deadline=deadline,
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
            # Keep HP/SP independent even when one fixed row is
            # temporarily unreadable. This is the recovery-critical path.
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

    values = read_status_panel(frame, origin=origin, deadline=deadline)
    if values is None:
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


def read_status_panel_snapshot_bounded(
    hwnd: int,
    confirmed: StatusPanelValues | None,
    *,
    refresh_max: bool,
    timeout_s: float = STATUS_PANEL_READ_TIMEOUT_S,
    client_hint: tuple[int, int, int, int] | None = None,
    refresh_client: bool = False,
    reanchor: bool = False,
) -> StatusPanelReadResult:
    """Run one panel read behind a hard caller-side timeout.

    The parser deadline protects its cooperative loops, while this daemon
    helper protects the UI work-queue thread from synchronous Win32 or native
    capture calls that cannot be interrupted from Python. A timed-out helper
    is intentionally abandoned; the single-flight guard prevents another
    native helper from being created until the abandoned call returns, and the
    stale result cannot publish through the feed generation guard.
    """
    # Keep the entire native read single-flight, including cached reads. The
    # UI capture channel already bounds mss.grab; this guard prevents a stuck
    # parser or unexpected native call from creating one abandoned OCR helper
    # per watchdog restart.
    if not _NATIVE_READ_IN_FLIGHT.acquire(blocking=False):
        return StatusPanelReadResult(
            hwnd=hwnd,
            state="read_timeout",
            error="previous native status read still in flight",
        )

    result_queue: queue.Queue[StatusPanelReadResult] = queue.Queue(maxsize=1)

    def _read() -> None:
        try:
            try:
                result = read_status_panel_snapshot(
                    hwnd,
                    confirmed,
                    refresh_max=refresh_max,
                    timeout_s=timeout_s,
                    client_hint=client_hint,
                    refresh_client=refresh_client,
                    reanchor=reanchor,
                )
            except Exception as exc:
                result = StatusPanelReadResult(
                    hwnd=hwnd,
                    state="read_failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            try:
                result_queue.put_nowait(result)
            except queue.Full:
                pass
        finally:
            _NATIVE_READ_IN_FLIGHT.release()

    try:
        threading.Thread(
            target=_read,
            name="ui-status-read-native",
            daemon=True,
        ).start()
    except Exception as exc:
        _NATIVE_READ_IN_FLIGHT.release()
        return StatusPanelReadResult(
            hwnd=hwnd,
            state="read_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    try:
        return result_queue.get(timeout=max(0.0, float(timeout_s)))
    except queue.Empty:
        return StatusPanelReadResult(
            hwnd=hwnd,
            state="read_timeout",
            error="native status read exceeded timeout",
        )
