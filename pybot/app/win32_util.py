"""Win32 helpers for window selection, focus, and game interaction.

"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

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


def is_window_minimized(hwnd: int) -> bool:
    """True when *hwnd* is minimized (iconic) and thus off-screen.

    Screen-grab OCR does not require the window to be the foreground window;
    it only requires the window to actually be rendered on screen. A minimized
    window is not on screen at all, so capturing its client rect would read
    whatever other content sits at those coordinates.
    """
    return bool(hwnd) and bool(user32.IsIconic(hwnd))
