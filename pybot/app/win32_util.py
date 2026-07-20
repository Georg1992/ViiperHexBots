"""Win32 helpers for window selection, focus, and game interaction.

"""

from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
gdi32 = ctypes.windll.gdi32

EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
SW_RESTORE = 9
SW_SHOWMINIMIZED = 2


class WINDOWPLACEMENT(ctypes.Structure):
    """Windows WINDOWPLACEMENT struct (not in ctypes.wintypes)."""
    _fields_ = [
        ("length", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("showCmd", wintypes.DWORD),
        ("ptMinPosition", wintypes.POINT),
        ("ptMaxPosition", wintypes.POINT),
        ("rcNormalPosition", wintypes.RECT),
    ]

# ── Overlay search region border windows ─────────────────────────

_search_overlay_hwnds: list[int] = []


@dataclass(frozen=True)
class WindowEntry:
    hwnd: int
    title: str
    process: str
    pid: int
    minimized: bool

    @property
    def display_text(self) -> str:
        # Include pid so two clients with the same title/exe stay distinct in
        # the combobox and memory reading can bind to the selected process.
        prefix = "[MIN] " if self.minimized else ""
        return f"{prefix}{self.title} ({self.process}) pid={self.pid}"


def _window_process_and_pid(hwnd: int) -> tuple[str, int]:
    process_id = wintypes.DWORD()
    tid = user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
    if not tid or process_id.value == 0:
        return "", 0
    pid = int(process_id.value)
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return "", pid
    try:
        buffer = ctypes.create_unicode_buffer(260)
        if kernel32.QueryFullProcessImageNameW(
            handle, 0, buffer, ctypes.byref(wintypes.DWORD(260))
        ):
            name = buffer.value.rsplit("\\", 1)[-1]
            return name, pid
    finally:
        kernel32.CloseHandle(handle)
    return "", pid


def enum_game_windows(*, exclude_hwnd: int = 0) -> list[WindowEntry]:
    entries: list[WindowEntry] = []

    def callback(hwnd: int, _lparam: int) -> bool:
        try:
            if hwnd == exclude_hwnd:
                return True
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            title_buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, title_buf, length + 1)
            title = title_buf.value.strip()
            process, pid = _window_process_and_pid(hwnd)
            if not title or not process or pid <= 0:
                return True
            if process.lower() == "explorer.exe":
                return True
            placement = WINDOWPLACEMENT()
            placement.length = ctypes.sizeof(placement)
            minimized = False
            if user32.GetWindowPlacement(hwnd, ctypes.byref(placement)):
                minimized = placement.showCmd == SW_SHOWMINIMIZED
            entries.append(
                WindowEntry(
                    hwnd=hwnd,
                    title=title,
                    process=process,
                    pid=pid,
                    minimized=minimized,
                )
            )
        except Exception:
            pass  # skip windows that cause enumeration errors
        return True

    if not user32.EnumWindows(EnumWindowsProc(callback), 0):
        # EnumWindows itself failed (extremely rare — callback always returns True).
        # Return whatever entries were collected before the failure.
        pass
    entries.sort(key=lambda item: item.display_text.lower())
    return entries


def window_exists(hwnd: int) -> bool:
    return bool(hwnd) and bool(user32.IsWindow(hwnd))


def client_rect_screen(hwnd: int) -> tuple[int, int, int, int] | None:
    """Return ``(left, top, width, height)`` of *hwnd*'s client area in screen coords."""
    if not window_exists(hwnd):
        return None
    client_rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(client_rect)):
        return None
    origin = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return None
    width = client_rect.right - client_rect.left
    height = client_rect.bottom - client_rect.top
    if width <= 0 or height <= 0:
        return None
    return int(origin.x), int(origin.y), int(width), int(height)


def restore_and_activate(hwnd: int) -> bool:
    """Restore (if minimised) and activate the target window.

    Returns immediately once GetForegroundWindow confirms the switch.
    Retries briefly for the rare case where Windows' foreground lock
    delays the switch (5 Ã 20 ms = 100 ms max).
    """
    if not window_exists(hwnd):
        return False
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    user32.SetForegroundWindow(hwnd)
    for _ in range(5):
        if user32.GetForegroundWindow() == hwnd:
            return True
        time.sleep(0.02)
    return user32.GetForegroundWindow() == hwnd


def is_window_active(hwnd: int) -> bool:
    if not window_exists(hwnd):
        return False
    active = user32.GetForegroundWindow()
    return active == hwnd


def flash_window_transparency(
    hwnd: int, *, alpha: int = 200, duration_ms: int = 120
) -> None:
    if not window_exists(hwnd):
        return
    user32.SetWindowLongW(
        hwnd, -20, user32.GetWindowLongW(hwnd, -20) | 0x80000
    )
    user32.SetLayeredWindowAttributes(hwnd, 0, alpha, 0x2)
    user32.SetTimer(hwnd, 0, duration_ms, None)


# ── Ported utility functions ──────────────────────────────────────


def set_default_keyboard_layout(layout: str = "00000409") -> None:
    """Set default keyboard layout (English US by default)."""
    hkl = user32.LoadKeyboardLayoutW(layout, 1)
    if hkl:
        user32.ActivateKeyboardLayout(hkl, 0)  # best-effort; fall back to system default on failure


def get_search_box_size_px(search_range: int, cell_size: int) -> int:
    """Calculate the total search box size in pixels."""
    return search_range * cell_size


def move_mouse_to(x: int, y: int) -> bool:
    """Move the cursor to screen coordinates."""
    return bool(user32.SetCursorPos(x, y))


def get_hunt_search_region(
    hwnd: int,
    search_range: int,
    cell_size: int,
) -> tuple[int, int, int, int] | None:
    """Calculate the centred hunt search region within the game client.

    GetHuntSearchRegion port.
    Returns (x, y, w, h) or None if the window is invalid.
    """
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

    from pybot.runtime.capture.window_roi import hunt_roi_from_client_rect

    roi = hunt_roi_from_client_rect(
        origin.x,
        origin.y,
        client_w,
        client_h,
        search_range_cells=search_range,
        cell_size_px=cell_size,
    )
    if roi is None:
        return None
    return roi.x, roi.y, roi.w, roi.h


def _create_colored_border_window(x: int, y: int, w: int, h: int, color: int) -> int:
    """Create a small popup window filled with a solid GDI colour."""
    hinstance = kernel32.GetModuleHandleW(None)

    hwnd = user32.CreateWindowExW(
        0x20 | 0x80000 | 0x8,  # WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOPMOST
        "Static",
        None,
        0x80000000,  # WS_POPUP only (no WS_VISIBLE yet)
        x, y, w, h,
        0, 0, hinstance, 0,
    )
    if not hwnd:
        return 0

    # Set a red background via window class brush (stock to avoid leak)
    brush = gdi32.GetStockObject(0)  # WHITE_BRUSH
    gdi32.SetClassLongPtrW(hwnd, -10, brush)  # GCL_HBRBACKGROUND = -10

    user32.SetLayeredWindowAttributes(hwnd, 0, 200, 0x02)
    user32.ShowWindow(hwnd, 8)  # SW_SHOWNA
    user32.InvalidateRect(hwnd, None, True)
    return hwnd


def show_search_region_overlay(
    x: int, y: int, w: int, h: int, duration_ms: int = 3000
) -> None:
    """Draw red border rectangles around the search region.

    ShowSearchRegionOverlay port.
    Uses four coloured border windows (top, bottom, left, right).
    """
    global _search_overlay_hwnds  # noqa: PLW0602
    hide_search_region_overlay()

    border = 3
    bottom_y = y + h - border
    right_x = x + w - border
    red = 0x000000FF  # GDI COLORREF (BGR format)

    _search_overlay_hwnds = [
        _create_colored_border_window(x, y, w, border, red),
        _create_colored_border_window(x, bottom_y, w, border, red),
        _create_colored_border_window(x, y, border, h, red),
        _create_colored_border_window(right_x, y, border, h, red),
    ]

    if duration_ms > 0:

        def _hide():
            time.sleep(duration_ms / 1000.0)
            hide_search_region_overlay()

        threading.Thread(target=_hide, daemon=True).start()


def hide_search_region_overlay() -> None:
    """Destroy all search region border windows."""
    global _search_overlay_hwnds  # noqa: PLW0602
    for hwnd in _search_overlay_hwnds:
        if hwnd and user32.IsWindow(hwnd):
            user32.DestroyWindow(hwnd)
    _search_overlay_hwnds = []


def check_image_on_screen(image_path: str) -> tuple[bool, int, int]:
    """Check if an image exists on screen (requires pyautogui + OpenCV).

    CheckImageOnScreen port.
    Returns (found, x, y).
    """
    try:
        import pyautogui
        pos = pyautogui.locateOnScreen(image_path, confidence=0.8)
        if pos is not None:
            return True, int(pos.left), int(pos.top)
    except ImportError:
        pass
    return False, 0, 0


def zoom_out(steps: int = 3, delay_ms: int = 200) -> None:
    """Zoom out in the game window by sending Ctrl+WheelDown.

    ZoomOut port.
    Uses SendInput for reliable input simulation.
    """
    from ctypes import Structure, c_ushort, c_ulong

    class MOUSEINPUT(Structure):
        _fields_ = [
            ("dx", c_ulong),
            ("dy", c_ulong),
            ("mouseData", c_ulong),
            ("dwFlags", c_ulong),
            ("time", c_ulong),
            ("dwExtraInfo", ctypes.c_void_p),
        ]

    class INPUT(Structure):
        _fields_ = [
            ("type", c_ulong),
            ("mi", MOUSEINPUT),
        ]

    INPUT_MOUSE = 0
    WHEEL_DOWN = 0xFF88  # One notch down (negative 120 as unsigned)

    for _ in range(steps):
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.mi.dwFlags = 0x0800  # MOUSEEVENTF_WHEEL
        inp.mi.mouseData = WHEEL_DOWN
        inserted = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        if not inserted:
            # SendInput can be blocked by UIPI (elevated process foreground lock)
            break
        time.sleep(delay_ms / 1000.0)
