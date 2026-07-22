"""Click-through overlay for Basic Info HP/SP/Weight vision.

Two states, same on-screen slot under Basic Info:
- Panel missing: prompt to open Basic Info
- Panel found: show current HP / SP / Weight

Stays visible during capture so the UI does not flash. Placement is below
the panel so the header and digit ROIs stay uncovered.
"""

from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes
from dataclasses import dataclass, field

from pybot.recognition.ui.status_panel import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    StatusPanelValues,
)

if not hasattr(wintypes, "LRESULT"):
    wintypes.LRESULT = (
        ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long
    )

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

WS_EX_LAYERED = 0x80000
WS_EX_TRANSPARENT = 0x20
WS_EX_TOOLWINDOW = 0x80
WS_EX_TOPMOST = 0x8
WS_POPUP = 0x80000000
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
WM_PAINT = 0x000F
WM_ERASEBKGND = 0x0014
WM_DESTROY = 0x0002
WDA_EXCLUDEFROMCAPTURE = 0x00000011

COLOR_BG = 0x001A1A1A
COLOR_TEXT = 0x00B8F0B8
COLOR_WARN = 0x00FFD966

LINE_H = 16
PAD_X = 6
PAD_Y = 4
VALUES_OVERLAY_H = 56
MESSAGE_OVERLAY_W = 340
PANEL_MISSING_LINES = (
    "Please Open Your Status Panel",
    "for visual detection",
)

user32.DefWindowProcW.argtypes = [
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
]
user32.DefWindowProcW.restype = wintypes.LRESULT


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", ctypes.c_void_p),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HCURSOR),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


class PAINTSTRUCT(ctypes.Structure):
    _fields_ = [
        ("hdc", wintypes.HDC),
        ("fErase", wintypes.BOOL),
        ("rcPaint", wintypes.RECT),
        ("fRestore", wintypes.BOOL),
        ("fIncUpdate", wintypes.BOOL),
        ("rgbReserved", wintypes.BYTE * 32),
    ]


@dataclass
class _State:
    hwnd: int = 0
    font: int = 0
    visible: bool = False
    paint_dirty: bool = False
    screen_x: int = 0
    screen_y: int = 0
    width: int = PANEL_WIDTH
    height: int = VALUES_OVERLAY_H
    warn: bool = False
    lines: tuple[str, ...] = ("HP —", "SP —", "Weight —")
    _lock: threading.Lock = field(default_factory=threading.Lock)


_state = _State()
WINDOW_CLASS = "StatusPanelOverlayClass"


def _create_font(name: str, size: int) -> int:
    return gdi32.CreateFontW(
        -size,
        0,
        0,
        0,
        400,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        name,
    )


def _wnd_proc(hwnd: int, msg: int, wparam: int, lparam: int) -> int:
    if msg == WM_ERASEBKGND:
        return 1
    if msg == WM_PAINT:
        ps = PAINTSTRUCT()
        hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))
        try:
            _paint(hdc)
        finally:
            user32.EndPaint(hwnd, ctypes.byref(ps))
        return 0
    if msg == WM_DESTROY:
        return 0
    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)


_WND_PROC_CALLBACK = ctypes.WINFUNCTYPE(
    wintypes.LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
)(_wnd_proc)


def _paint(hdc: int) -> None:
    with _state._lock:
        width = _state.width
        height = _state.height
        lines = _state.lines
        warn = _state.warn
        font = _state.font
    brush = gdi32.CreateSolidBrush(COLOR_BG)
    rect = wintypes.RECT(0, 0, width, height)
    user32.FillRect(hdc, ctypes.byref(rect), brush)
    gdi32.DeleteObject(brush)
    if not font:
        return
    old = gdi32.SelectObject(hdc, font)
    gdi32.SetBkMode(hdc, 1)
    gdi32.SetTextColor(hdc, COLOR_WARN if warn else COLOR_TEXT)
    y = PAD_Y
    for line in lines:
        gdi32.TextOutW(hdc, PAD_X, y, line, len(line))
        y += LINE_H
    gdi32.SelectObject(hdc, old)


def create() -> bool:
    """Create the overlay window if needed."""
    if _state.hwnd and user32.IsWindow(_state.hwnd):
        return True
    _state.hwnd = 0
    hinstance = kernel32.GetModuleHandleW(None)
    cls = WNDCLASSW()
    cls.lpfnWndProc = ctypes.cast(_WND_PROC_CALLBACK, ctypes.c_void_p)
    cls.hInstance = hinstance
    cls.hbrBackground = 0
    cls.lpszClassName = WINDOW_CLASS
    user32.RegisterClassW(ctypes.byref(cls))
    hwnd = user32.CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_TOPMOST,
        WINDOW_CLASS,
        "StatusPanelOverlay",
        WS_POPUP,
        0,
        0,
        PANEL_WIDTH,
        VALUES_OVERLAY_H,
        0,
        0,
        hinstance,
        0,
    )
    if not hwnd:
        return False
    _state.hwnd = hwnd
    _state.font = _create_font("Consolas", 13)
    if not _state.font:
        destroy()
        return False
    user32.SetLayeredWindowAttributes(hwnd, 0, 230, 0x02)
    try:
        user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
    except AttributeError:
        pass
    _state.visible = False
    return True


def destroy() -> None:
    hwnd = _state.hwnd
    font = _state.font
    _state.hwnd = 0
    _state.font = 0
    _state.visible = False
    if font:
        gdi32.DeleteObject(font)
    if hwnd and user32.IsWindow(hwnd):
        user32.DestroyWindow(hwnd)


def hide() -> None:
    with _state._lock:
        _state.visible = False
        hwnd = _state.hwnd
    if hwnd and user32.IsWindow(hwnd):
        user32.ShowWindow(hwnd, 0)


def _show_at(
    *,
    screen_x: int,
    screen_y: int,
    width: int,
    height: int,
    lines: tuple[str, ...],
    warn: bool,
) -> None:
    if not create():
        return
    with _state._lock:
        same_geom = (
            _state.visible
            and _state.screen_x == screen_x
            and _state.screen_y == screen_y
            and _state.width == width
            and _state.height == height
        )
        same_content = _state.lines == lines and _state.warn == warn
        if same_geom and same_content:
            return
        _state.screen_x = screen_x
        _state.screen_y = screen_y
        _state.width = width
        _state.height = height
        _state.lines = lines
        _state.warn = warn
        _state.visible = True
        hwnd = _state.hwnd
        need_move = not same_geom
        need_paint = not same_content or not same_geom
    if need_move:
        user32.SetWindowPos(
            hwnd,
            0,
            screen_x,
            screen_y,
            width,
            height,
            SWP_NOZORDER | SWP_NOACTIVATE,
        )
        user32.ShowWindow(hwnd, 8)  # SW_SHOWNA
    elif not user32.IsWindowVisible(hwnd):
        user32.ShowWindow(hwnd, 8)
    if need_paint:
        _state.paint_dirty = True
        _flush_paint()


def update(
    values: StatusPanelValues,
    *,
    client_left: int,
    client_top: int,
) -> None:
    """Show parsed values under the panel."""
    _show_under_panel(
        client_left=client_left,
        client_top=client_top,
        panel_origin=values.panel_origin,
        width=PANEL_WIDTH,
        height=VALUES_OVERLAY_H,
        lines=(
            f"HP {values.hp}/{values.hp_max}",
            f"SP {values.sp}/{values.sp_max}",
            f"Weight {_format_weight(values)}",
        ),
        warn=False,
    )


def show_panel_missing(
    *,
    client_left: int,
    client_top: int,
    panel_origin: tuple[int, int] = (0, 0),
) -> None:
    """Prompt to open Basic Info — same slot as the HP/SP/Weight overlay."""
    height = PAD_Y * 2 + LINE_H * len(PANEL_MISSING_LINES)
    _show_under_panel(
        client_left=client_left,
        client_top=client_top,
        panel_origin=panel_origin,
        width=MESSAGE_OVERLAY_W,
        height=height,
        lines=PANEL_MISSING_LINES,
        warn=True,
    )


def _format_weight(values: StatusPanelValues) -> str:
    if values.weight is None:
        return "—"
    if values.weight_max is not None:
        return f"{values.weight}/{values.weight_max}"
    return str(values.weight)


def _show_under_panel(
    *,
    client_left: int,
    client_top: int,
    panel_origin: tuple[int, int],
    width: int,
    height: int,
    lines: tuple[str, ...],
    warn: bool,
) -> None:
    ox, oy = panel_origin
    _show_at(
        screen_x=client_left + ox,
        screen_y=client_top + oy + PANEL_HEIGHT + 2,
        width=width,
        height=height,
        lines=lines,
        warn=warn,
    )


def _flush_paint() -> None:
    if not _state.paint_dirty:
        return
    _state.paint_dirty = False
    hwnd = _state.hwnd
    if hwnd and user32.IsWindow(hwnd):
        user32.InvalidateRect(hwnd, None, False)
        user32.UpdateWindow(hwnd)


class StatusPanelOverlay:
    """Injectable wrapper for the status-panel values overlay."""

    def create(self) -> bool:
        return create()

    def destroy(self) -> None:
        destroy()

    def hide(self) -> None:
        hide()

    def update(
        self,
        values: StatusPanelValues,
        *,
        client_left: int,
        client_top: int,
    ) -> None:
        update(values, client_left=client_left, client_top=client_top)

    def show_panel_missing(
        self,
        *,
        client_left: int,
        client_top: int,
        panel_origin: tuple[int, int] = (0, 0),
    ) -> None:
        show_panel_missing(
            client_left=client_left,
            client_top=client_top,
            panel_origin=panel_origin,
        )
