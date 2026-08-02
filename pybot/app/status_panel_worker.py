"""Background status-panel capture/OCR helpers.

This module deliberately contains no Tk calls.  MainWindow starts one short
worker at a time and applies the returned immutable result on the UI thread.
"""

from __future__ import annotations

from dataclasses import dataclass

from pybot.app.win32_util import client_rect_screen, is_window_active, window_exists
from pybot.recognition.capture import capture_region
from pybot.recognition.ui.status_panel import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    StatusPanelValues,
    find_status_panel,
    read_status_panel,
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
    full_refresh: bool = False


def read_status_panel_snapshot(
    hwnd: int,
    confirmed: StatusPanelValues | None,
    *,
    refresh_max: bool,
) -> StatusPanelReadResult:
    """Capture and parse one panel snapshot without touching Tk."""
    if not hwnd or not window_exists(hwnd) or not is_window_active(hwnd):
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
            panel_frame = capture_region(
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
                        return StatusPanelReadResult(
                            hwnd=hwnd,
                            state="panel_open_digits_missing",
                            client_left=left,
                            client_top=top,
                            client_width=width,
                            client_height=height,
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

    frame = capture_region(left, top, width, height)
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
        return StatusPanelReadResult(
            hwnd=hwnd,
            state="panel_open_digits_missing",
            client_left=left,
            client_top=top,
            client_width=width,
            client_height=height,
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
