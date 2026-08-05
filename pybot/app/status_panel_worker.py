"""Background status-panel capture/OCR helpers.

This module deliberately contains no Tk calls.  MainWindow starts one short
worker at a time and applies the returned immutable result on the UI thread.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from pybot.app.win32_util import client_rect_screen, is_window_minimized, window_exists
from pybot.recognition.capture import ui_capture_region
from pybot.recognition.ui.status_panel import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    StatusPanelValues,
    find_status_panel,
    read_status_panel,
    read_status_panel_hp,
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
    full_refresh: bool = False


def read_status_panel_snapshot(
    hwnd: int,
    confirmed: StatusPanelValues | None,
    *,
    refresh_max: bool,
) -> StatusPanelReadResult:
    """Capture and parse one panel snapshot without touching Tk.

    OCR reads screen pixels via mss, so it works whenever the game window is
    rendered on screen — it must NOT depend on the window being the foreground
    window. The only truly unreadable state is a minimized (off-screen) window.
    """
    if not hwnd or not window_exists(hwnd) or is_window_minimized(hwnd):
        # "inactive" here means unreadable (window gone or minimized/off-screen),
        # NOT "not the foreground window" — foreground status must never gate OCR.
        return StatusPanelReadResult(hwnd=hwnd, state="inactive")

    client = client_rect_screen(hwnd)
    if client is None:
        return StatusPanelReadResult(hwnd=hwnd, state="client_missing")

    left, top, width, height = client
    if not refresh_max and confirmed is not None:
        ox, oy = confirmed.panel_origin
        panel_in_client = (
            ox >= 0
            and oy >= 0
            and ox + PANEL_WIDTH <= width
            and oy + PANEL_HEIGHT <= height
        )
        if panel_in_client:
            panel_frame = ui_capture_region(
                left + ox, top + oy, PANEL_WIDTH, PANEL_HEIGHT
            )
            if panel_frame is not None and panel_frame.size > 0:
                if verify_status_panel_at(panel_frame, (0, 0)):
                    values = read_status_panel(
                        panel_frame,
                        origin=(0, 0),
                        skip_hp=True,
                        previous=confirmed,
                    )
                    if values is None:
                        hp = read_status_panel_hp(panel_frame, origin=(0, 0))
                        return StatusPanelReadResult(
                            hwnd=hwnd,
                            state="hp_only" if hp is not None else "panel_open_digits_missing",
                            client_left=left,
                            client_top=top,
                            client_width=width,
                            client_height=height,
                            hp=hp,
                        )
                    return StatusPanelReadResult(
                        hwnd=hwnd,
                        state="values",
                        client_left=left,
                        client_top=top,
                        client_width=width,
                        client_height=height,
                        values=StatusPanelValues(
                            hp=values.hp,
                            hp_max=values.hp_max,
                            sp=values.sp,
                            sp_max=values.sp_max,
                            weight=values.weight,
                            weight_max=values.weight_max,
                            panel_origin=(ox, oy),
                        ),
                    )
        # The locked origin moved or became invalid; locate it again below.

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

    origin = find_status_panel(frame)
    if origin is None:
        return StatusPanelReadResult(
            hwnd=hwnd,
            state="panel_missing",
            client_left=left,
            client_top=top,
            client_width=width,
            client_height=height,
        )

    values = read_status_panel(frame, origin=origin)
    if values is None:
        hp = read_status_panel_hp(frame, origin=origin)
        return StatusPanelReadResult(
            hwnd=hwnd,
            state="hp_only" if hp is not None else "panel_open_digits_missing",
            client_left=left,
            client_top=top,
            client_width=width,
            client_height=height,
            hp=hp,
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


STATUS_PANEL_READ_TIMEOUT_S = 5.0


def read_status_panel_snapshot_bounded(
    hwnd: int,
    confirmed: StatusPanelValues | None,
    *,
    refresh_max: bool,
    timeout_s: float = STATUS_PANEL_READ_TIMEOUT_S,
) -> StatusPanelReadResult:
    """Run one panel read with a hard caller-side timeout.

    ``GetClientRect``/``ClientToScreen`` are synchronous Win32 calls and can
    stop returning while a game client is wedged during a teleport/loading
    transition. Running the existing read on a daemon helper prevents that
    native call from pinning the UI work-queue thread forever. The helper is
    intentionally abandoned on timeout; the next call gets a fresh helper.
    """
    result_queue: queue.Queue[StatusPanelReadResult] = queue.Queue(maxsize=1)

    def _read() -> None:
        try:
            result = read_status_panel_snapshot(
                hwnd, confirmed, refresh_max=refresh_max
            )
        except BaseException:
            # Preserve the existing caller-side exception handling contract by
            # returning a synthetic failure rather than blocking for a native
            # call that may never unwind.
            result = StatusPanelReadResult(hwnd=hwnd, state="read_failed")
        try:
            result_queue.put_nowait(result)
        except queue.Full:
            pass

    threading.Thread(target=_read, name="ui-status-read-native", daemon=True).start()
    try:
        return result_queue.get(timeout=max(0.0, timeout_s))
    except queue.Empty:
        return StatusPanelReadResult(hwnd=hwnd, state="read_timeout")
