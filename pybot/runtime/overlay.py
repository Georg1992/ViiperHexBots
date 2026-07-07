"""Transparent click-through hunt overlay on the game client window.

Creates a full-client-area layered window over the game showing:
- Colored dots at tracked mob positions (green=attackable,
  yellow=pending, orange=non-attackable alive)
- A dark right-side panel with track stats + scrolling log.
The dots render on a transparent overlay and never appear in
captured game frames, so detection is not affected.
Repositions every ~400 ms via a timer thread.
"""

from __future__ import annotations

import ctypes
import re
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass, field

# LRESULT was removed from wintypes in Python 3.14
if not hasattr(wintypes, "LRESULT"):
    wintypes.LRESULT = ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long

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
WM_TIMER = 0x0113

# Explicit argtypes for DefWindowProcW so 64-bit WPARAM/LPARAM don't overflow
user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.DefWindowProcW.restype = wintypes.LRESULT

# Colour key for transparent game-area pixels (hot pink so it's never used by game UI)
COLOR_KEY = 0x00FF00FF  # BGR magenta

COLOR_BLACK = 0x001A1A1A
COLOR_TEXT = 0x00B8F0B8
COLOR_STATUS = 0x00FFD966

# Dot colours (GDI COLORREF = BGR)
COLOR_DOT_LIVING = 0x0000FF00      # green — attackable
COLOR_DOT_PENDING = 0x0000FFFF     # yellow — pending result
COLOR_DOT_ALIVE = 0x0000A5FF       # orange — alive, not attackable

# Search region border colour
COLOR_ROI = 0x00FFE066  # light amber — visible but unobtrusive

PANEL_W = 300  # right-side panel width for status/log


class WNDCLASSW(ctypes.Structure):
    """Windows WNDCLASSW struct (not in ctypes.wintypes)."""
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
    """Windows PAINTSTRUCT struct (not in ctypes.wintypes)."""
    _fields_ = [
        ("hdc", wintypes.HDC),
        ("fErase", wintypes.BOOL),
        ("rcPaint", wintypes.RECT),
        ("fRestore", wintypes.BOOL),
        ("fIncUpdate", wintypes.BOOL),
        ("rgbReserved", wintypes.BYTE * 32),
    ]


@dataclass
class _OverlayState:
    hwnd: int = 0
    font_status: int = 0
    font_log: int = 0
    visible: bool = False
    last_scan_living: int = 0
    track_count: int = 0
    alive_or_pending_count: int = 0
    attackable_count: int = 0
    total_attacks: int = 0
    total_kills: int = 0
    total_teleports: int = 0
    track_positions: list[tuple[int, int, str]] = field(default_factory=list)
    client_left: int = 0  # screen X of game client origin
    client_top: int = 0   # screen Y of game client origin
    client_w: int = 0     # game client width (for panel calc)
    brush_living: int = 0
    brush_pending: int = 0
    brush_alive: int = 0
    brush_roi: int = 0
    roi_x: int = 0
    roi_y: int = 0
    roi_w: int = 0
    roi_h: int = 0
    log_lines: list[str] = field(default_factory=list)
    running: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)
    game_hwnd: int = 0


_state = _OverlayState()
_last_create_error: str = ""

_HUNT_LINE_RE = re.compile(
    r"^(\[(?:HUNT|TRACK|DISCOVERY|STATE|DIRECT|MODE)\])",
    re.IGNORECASE,
)
_ALSO_LINES = re.compile(
    r"^(Bot (?:started|stopped|paused|resumed)|WARNING:)", re.IGNORECASE
)


def _is_hunt_line(message: str) -> bool:
    return bool(_HUNT_LINE_RE.match(message) or _ALSO_LINES.match(message))


def _get_client_rect_screen(hwnd: int) -> tuple[int, int, int, int] | None:
    if not hwnd or not user32.IsWindow(hwnd):
        return None
    client_rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(client_rect)):
        return None
    origin = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return None
    cw = client_rect.right - client_rect.left
    ch = client_rect.bottom - client_rect.top
    if cw <= 0 or ch <= 0:
        return None
    return origin.x, origin.y, cw, ch


def _create_font(name: str, height: int) -> int:
    return gdi32.CreateFontW(height, 0, 0, 0, 400, 0, 0, 0, 0, 0, 0, 0, 0, name)


# ── Window procedure ──────────────────────────────────────────────

WndProcPtr = ctypes.WINFUNCTYPE(
    wintypes.LRESULT, wintypes.HWND, wintypes.UINT,
    wintypes.WPARAM, wintypes.LPARAM,
)


def _wnd_proc(hwnd: int, msg: int, _wparam: int, _lparam: int) -> int:
    if msg == WM_PAINT:
        ps = PAINTSTRUCT()
        hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))
        if hdc:
            _paint_overlay(hdc)
            user32.EndPaint(hwnd, ctypes.byref(ps))
        return 0
    elif msg == WM_ERASEBKGND:
        return 1
    elif msg == WM_DESTROY:
        _state.running = False
        return 0
    return user32.DefWindowProcW(hwnd, msg, _wparam, _lparam)


_WND_PROC_CALLBACK = WndProcPtr(_wnd_proc)


def _paint_overlay(hdc: int) -> None:
    s = _state
    if not s.hwnd:
        return
    rect = wintypes.RECT()
    user32.GetClientRect(s.hwnd, ctypes.byref(rect))
    cw = rect.right - rect.left
    ch = rect.bottom - rect.top

    # ── Game-area transparent fill ─────────────────────────────
    # Fill the entire window with COLOR_KEY (magenta).  The layered
    # window uses LWA_COLORKEY so these pixels are fully transparent
    # on screen, making the game visible through the overlay.
    # Everything drawn AFTER this (dots, panel, text) uses other
    # colours so it renders at the alpha set by LWA_ALPHA.
    key_brush = gdi32.CreateSolidBrush(COLOR_KEY)
    full_rect = wintypes.RECT(0, 0, cw, ch)
    user32.FillRect(hdc, ctypes.byref(full_rect), key_brush)
    gdi32.DeleteObject(key_brush)

    # ── Search region border ──────────────────────────────────
    if s.roi_w > 0 and s.brush_roi and s.client_w > 0:
        rx = s.roi_x - s.client_left
        ry = s.roi_y - s.client_top
        roi_rect = wintypes.RECT(rx, ry, rx + s.roi_w, ry + s.roi_h)
        user32.FrameRect(hdc, ctypes.byref(roi_rect), s.brush_roi)

    # ── Draw track position dots (over the game area) ──────────
    if s.track_positions and s.client_w > 0:
        for tx, ty, state_type in s.track_positions:
            # Convert game-client-absolute coords to overlay-relative
            dx = tx - s.client_left
            dy = ty - s.client_top
            # Skip if inside the right-side panel
            if dx >= cw - PANEL_W:
                continue
            if state_type == "attackable":
                brush = s.brush_living
            elif state_type == "pending":
                brush = s.brush_pending
            else:
                brush = s.brush_alive
            if brush:
                old_b = gdi32.SelectObject(hdc, brush)
                gdi32.Ellipse(hdc, dx - 4, dy - 4, dx + 4, dy + 4)
                gdi32.SelectObject(hdc, old_b)

    # ── Right-side panel background ────────────────────────────
    panel_rect = wintypes.RECT(
        max(cw - PANEL_W, 0), 0, cw, ch
    )
    brush_bg = gdi32.CreateSolidBrush(COLOR_BLACK)
    user32.FillRect(hdc, ctypes.byref(panel_rect), brush_bg)
    gdi32.DeleteObject(brush_bg)

    # ── Status line ────────────────────────────────────────────
    old_font = gdi32.SelectObject(hdc, s.font_status)
    gdi32.SetTextColor(hdc, COLOR_STATUS)
    gdi32.SetBkMode(hdc, 1)  # TRANSPARENT
    # Left-align panel content with 6px padding
    px = cw - PANEL_W + 6 if cw > PANEL_W else 6
    status = f"T:{s.track_count} A:{s.alive_or_pending_count} Atk:{s.attackable_count} K:{s.total_kills} TP:{s.total_teleports}"
    gdi32.TextOutW(hdc, px, 6, status, len(status))

    # ── Log lines ──────────────────────────────────────────────
    gdi32.SelectObject(hdc, s.font_log)
    gdi32.SetTextColor(hdc, COLOR_TEXT)
    y = 30
    max_lines = (ch - 34) // 17
    with s._lock:
        lines = list(s.log_lines)
    for line in lines[-max_lines:]:
        gdi32.TextOutW(hdc, px, y, line, len(line))
        y += 17
    gdi32.SelectObject(hdc, old_font)


def _reposition() -> None:
    if not _state.hwnd or not _state.visible:
        return
    client = _get_client_rect_screen(_state.game_hwnd)
    if client is None:
        destroy()
        return
    client_left, client_top, client_w, client_h = client
    _state.client_left = client_left
    _state.client_top = client_top
    _state.client_w = client_w
    # Full-game overlay — covers the entire game client
    user32.SetWindowPos(
        _state.hwnd, 0, client_left, client_top, client_w, client_h,
        SWP_NOZORDER | SWP_NOACTIVATE,
    )


# ── Public API ────────────────────────────────────────────────────

WINDOW_CLASS = "HuntOverlayClass"


def _create_brushes() -> tuple[int, int, int, int]:
    """Create solid GDI brushes for dots and ROI border."""
    return (
        gdi32.CreateSolidBrush(COLOR_DOT_LIVING),
        gdi32.CreateSolidBrush(COLOR_DOT_PENDING),
        gdi32.CreateSolidBrush(COLOR_DOT_ALIVE),
        gdi32.CreateSolidBrush(COLOR_ROI),
    )


def create(game_hwnd: int) -> bool:
    """Create (or recreate) the overlay window over *game_hwnd*."""
    global _last_create_error

    if _state.hwnd:
        # Already created — verify the window is still valid
        if user32.IsWindow(_state.hwnd):
            return True
        # Stale handle from a previous session, reset
        _state.hwnd = 0

    _state.game_hwnd = game_hwnd
    _last_create_error = ""

    client = _get_client_rect_screen(game_hwnd)
    if client is None:
        _last_create_error = f"get_client_rect_screen failed for hwnd={game_hwnd}"
        return False

    _client_left, _client_top, client_w, client_h = client

    hinstance = kernel32.GetModuleHandleW(None)

    cls = WNDCLASSW()
    cls.lpfnWndProc = ctypes.cast(_WND_PROC_CALLBACK, ctypes.c_void_p)
    cls.hInstance = hinstance
    cls.hbrBackground = 0
    cls.lpszClassName = WINDOW_CLASS
    user32.RegisterClassW(ctypes.byref(cls))

    hwnd = user32.CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_TOPMOST,
        WINDOW_CLASS, "HuntOverlay", WS_POPUP,
        0, 0, client_w, client_h,
        0, 0, hinstance, 0,
    )
    if not hwnd:
        _last_create_error = f"CreateWindowExW failed for size={client_w}x{client_h}"
        return False

    _state.hwnd = hwnd
    _state.font_status = _create_font("Consolas", 14)
    _state.font_log = _create_font("Consolas", 12)
    (
        _state.brush_living,
        _state.brush_pending,
        _state.brush_alive,
        _state.brush_roi,
    ) = _create_brushes()
    if not _state.font_status or not _state.font_log:
        _last_create_error = "CreateFontW failed"
        destroy()
        return False
    # Layered-window attributes:
    #   LWA_COLORKEY (0x01) — pixels matching COLOR_KEY are fully transparent
    #   LWA_ALPHA    (0x02) — other pixels get 220/255 opacity (~86%)
    #   Combined 0x03        — game area (magenta) invisible, panel stays semi-transparent
    user32.SetLayeredWindowAttributes(hwnd, COLOR_KEY, 220, 0x03)

    # Hide this window from screen-capture APIs (mss, BitBlt, etc.)
    # so the bot's detection pipeline never sees the overlay
    WDA_EXCLUDEFROMCAPTURE = 0x00000011
    try:
        user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
    except AttributeError:
        # Pre-Windows 10 2004 — ignore, capture will include overlay
        pass

    _state.visible = True
    _reposition()
    user32.ShowWindow(hwnd, 8)  # SW_SHOWNA
    # Force an immediate initial paint
    _invalidate()

    return True


def last_error() -> str:
    """Return the last overlay creation error message."""
    return _last_create_error


def _destroy_brushes() -> None:
    for attr in ("brush_living", "brush_pending", "brush_alive", "brush_roi"):
        brush = getattr(_state, attr, 0)
        if brush:
            gdi32.DeleteObject(brush)
            setattr(_state, attr, 0)


def destroy() -> None:
    """Destroy the overlay window and release GDI resources."""
    _state.running = False
    if _state.hwnd and user32.IsWindow(_state.hwnd):
        user32.KillTimer(_state.hwnd, 1)
        user32.DestroyWindow(_state.hwnd)
    _state.hwnd = 0
    _state.visible = False
    _state.log_lines.clear()
    _state.track_positions.clear()
    if _state.font_status:
        gdi32.DeleteObject(_state.font_status)
        _state.font_status = 0
    if _state.font_log:
        gdi32.DeleteObject(_state.font_log)
        _state.font_log = 0
    _destroy_brushes()


def append_log(timestamped_line: str, raw_message: str) -> None:
    """Add a log line to the overlay if it matches hunt patterns."""
    if not _is_hunt_line(raw_message):
        return
    with _state._lock:
        if _state.log_lines and _state.log_lines[-1] == timestamped_line:
            return
        _state.log_lines.append(timestamped_line)
        if len(_state.log_lines) > 24:
            _state.log_lines.pop(0)
    if _state.hwnd and user32.IsWindow(_state.hwnd):
        user32.InvalidateRect(_state.hwnd, None, True)


def set_scan_living(count: int) -> None:
    """Update the 'scan living' count shown in the overlay status line."""
    _state.last_scan_living = count
    _invalidate()


def set_track_stats(
    track_count: int,
    alive_count: int,
    attackable_count: int,
) -> None:
    """Update track stats shown in the overlay status line."""
    _state.track_count = track_count
    _state.alive_or_pending_count = alive_count
    _state.attackable_count = attackable_count
    _invalidate()


def increment_attacks() -> None:
    """Increment the total attacks counter."""
    _state.total_attacks += 1
    _invalidate()


def increment_teleports() -> None:
    """Increment the total teleports counter."""
    _state.total_teleports += 1
    _invalidate()


def increment_kills() -> None:
    """Increment the total kills counter."""
    _state.total_kills += 1
    _invalidate()


def set_track_positions(
    positions: list[tuple[int, int, str]],
) -> None:
    """Update tracked mob positions for dot rendering.

    Each entry is ``(screen_x, screen_y, state_type)`` where
    *state_type* is one of ``"attackable"``, ``"pending"``, or
    ``"alive"``.
    """
    _state.track_positions = positions
    _invalidate()


def set_search_roi(x: int, y: int, w: int, h: int) -> None:
    """Set the search region rectangle to draw on the overlay."""
    _state.roi_x = x
    _state.roi_y = y
    _state.roi_w = w
    _state.roi_h = h
    _invalidate()


def reset_stats() -> None:
    """Reset all counters and positions (call when a new bot session starts)."""
    _state.last_scan_living = 0
    _state.track_count = 0
    _state.alive_or_pending_count = 0
    _state.attackable_count = 0
    _state.total_attacks = 0
    _state.total_kills = 0
    _state.total_teleports = 0
    _state.track_positions.clear()
    _invalidate()


def _invalidate() -> None:
    """Force an immediate repaint (synchronous, works cross-thread).

    Uses ``UpdateWindow`` which sends WM_PAINT synchronously via
    ``SendMessage`` — this works even when called from a thread that
    didn't create the overlay window (unlike the message-queue-based
    approach that would require pumping messages on the right thread).
    """
    if _state.hwnd and user32.IsWindow(_state.hwnd):
        user32.InvalidateRect(_state.hwnd, None, True)
        user32.UpdateWindow(_state.hwnd)


def tick() -> None:
    """Periodic upkeep: reposition overlay and repaint.

    Called from the tkinter UI's ``after()`` loop (~400 ms interval).
    Replaces the previous ``SetTimer`` + message-pump-thread approach
    which was broken because posted messages are only delivered to the
    thread that created the window (the main thread), not the pump thread.
    """
    if _state.visible:
        _reposition()
        _invalidate()
