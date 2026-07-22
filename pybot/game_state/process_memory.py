"""Read game stats from a client process via module-relative offsets.

Offsets in ``clients/*.json`` are relative to the exe module base (ASLR-safe).
Each poll opens a fresh ``PROCESS_VM_READ`` handle, reads name/SP/weight, then
closes the handle — same pattern as the Belarus Champ Tools address reader.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass

from pybot.config.clients import MemoryAddresses

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

PROCESS_VM_READ = 0x0010
TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.ReadProcessMemory.argtypes = [
    wintypes.HANDLE,
    wintypes.LPCVOID,
    wintypes.LPVOID,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
kernel32.ReadProcessMemory.restype = wintypes.BOOL
kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
kernel32.Module32FirstW.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
kernel32.Module32FirstW.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = [
    wintypes.HWND,
    ctypes.POINTER(wintypes.DWORD),
]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD


class MODULEENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("th32ModuleID", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("GlblcntUsage", wintypes.DWORD),
        ("ProccntUsage", wintypes.DWORD),
        ("modBaseAddr", ctypes.c_void_p),
        ("modBaseSize", wintypes.DWORD),
        ("hModule", wintypes.HMODULE),
        ("szModule", ctypes.c_wchar * 256),
        ("szExePath", ctypes.c_wchar * 260),
    ]


# RO names are short; read a small buffer and cut at the first NUL.
CHAR_NAME_MAX_BYTES = 32


@dataclass(frozen=True)
class MemorySnapshot:
    char_name: str | None = None
    hp: int | None = None
    hp_max: int | None = None
    sp: int | None = None
    sp_max: int | None = None
    weight: int | None = None
    weight_max: int | None = None
    ok: bool = False
    error: str = ""


def pid_from_hwnd(hwnd: int) -> int:
    if not hwnd:
        return 0
    process_id = wintypes.DWORD()
    tid = user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
    if not tid:
        return 0
    return int(process_id.value)


def module_base_address(pid: int) -> int:
    """Base address of the first module (the executable) for *pid*."""
    if pid <= 0:
        raise OSError("invalid pid")
    flags = TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32
    snapshot = kernel32.CreateToolhelp32Snapshot(flags, pid)
    if snapshot == INVALID_HANDLE_VALUE or not snapshot:
        raise OSError(f"CreateToolhelp32Snapshot failed for pid={pid}")
    try:
        entry = MODULEENTRY32W()
        entry.dwSize = ctypes.sizeof(MODULEENTRY32W)
        if not kernel32.Module32FirstW(snapshot, ctypes.byref(entry)):
            raise OSError(f"Module32FirstW failed for pid={pid}")
        base = entry.modBaseAddr
        if not base:
            raise OSError(f"empty module base for pid={pid}")
        return int(base)
    finally:
        kernel32.CloseHandle(snapshot)


def _read_uint32(handle: int, absolute_addr: int) -> int:
    value = wintypes.DWORD()
    n_read = ctypes.c_size_t(0)
    ok = kernel32.ReadProcessMemory(
        handle,
        ctypes.c_void_p(absolute_addr),
        ctypes.byref(value),
        ctypes.sizeof(value),
        ctypes.byref(n_read),
    )
    if not ok or n_read.value != ctypes.sizeof(value):
        raise OSError(f"ReadProcessMemory failed at 0x{absolute_addr:X}")
    return int(value.value)


def _read_cstring(handle: int, absolute_addr: int, *, max_bytes: int = CHAR_NAME_MAX_BYTES) -> str:
    buf = (ctypes.c_char * max_bytes)()
    n_read = ctypes.c_size_t(0)
    ok = kernel32.ReadProcessMemory(
        handle,
        ctypes.c_void_p(absolute_addr),
        ctypes.byref(buf),
        max_bytes,
        ctypes.byref(n_read),
    )
    if not ok or n_read.value == 0:
        raise OSError(f"ReadProcessMemory string failed at 0x{absolute_addr:X}")
    raw = bytes(buf)[: n_read.value]
    text = raw.split(b"\x00", 1)[0]
    return text.decode("latin-1", errors="replace").strip()


def read_snapshot(
    pid: int,
    module_base: int,
    addresses: MemoryAddresses,
) -> MemorySnapshot:
    """Read name/SP/weight for *pid* using *module_base* + profile offsets."""
    if pid <= 0:
        return MemorySnapshot(error="no process")
    if module_base <= 0:
        return MemorySnapshot(error="no module base")
    if not addresses.has_any:
        return MemorySnapshot(error="no addresses")

    handle = kernel32.OpenProcess(PROCESS_VM_READ, False, pid)
    if not handle:
        return MemorySnapshot(error="OpenProcess failed")
    try:
        def optional_u32(offset: int) -> int | None:
            if not offset:
                return None
            return _read_uint32(handle, module_base + offset)

        def optional_name(offset: int) -> str | None:
            if not offset:
                return None
            name = _read_cstring(handle, module_base + offset)
            return name or None

        return MemorySnapshot(
            char_name=optional_name(addresses.char_name),
            sp=optional_u32(addresses.current_sp),
            sp_max=optional_u32(addresses.max_sp),
            weight=optional_u32(addresses.current_weight),
            weight_max=optional_u32(addresses.max_weight),
            ok=True,
        )
    except OSError as exc:
        return MemorySnapshot(error=str(exc))
    finally:
        kernel32.CloseHandle(handle)


def _client_rect_screen(hwnd: int) -> tuple[int, int, int, int] | None:
    """``(left, top, width, height)`` of *hwnd*'s client area in screen coords."""
    if not hwnd or not user32.IsWindow(hwnd):
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


def read_vision_snapshot(hwnd: int) -> MemorySnapshot:
    """Read HP/SP/Weight from the open Basic Info panel into a ``MemorySnapshot``.

    HP is vision-only (never filled from process memory). ``char_name`` is
    not available from vision.
    """
    from pybot.recognition.capture import capture_region
    from pybot.recognition.ui.status_panel import read_status_panel

    if not hwnd:
        return MemorySnapshot(error="select a game window")
    client = _client_rect_screen(hwnd)
    if client is None:
        return MemorySnapshot(error="no client rect")
    left, top, width, height = client
    frame = capture_region(left, top, width, height)
    if frame is None or frame.size == 0:
        return MemorySnapshot(error="capture_empty")
    values = read_status_panel(frame)
    if values is None:
        return MemorySnapshot(error="status_panel_unreadable")
    return MemorySnapshot(
        hp=values.hp,
        hp_max=values.hp_max,
        sp=values.sp,
        sp_max=values.sp_max,
        weight=values.weight,
        weight_max=values.weight_max,
        ok=True,
    )


class GameMemoryPoller:
    """Produces ``MemorySnapshot`` from process memory or status-panel vision.

    When *addresses* include SP offsets, reads memory. Otherwise (Generic)
    OCR's the Basic Info panel into the same snapshot fields.
    """

    def __init__(self) -> None:
        self._pid = 0
        self._base = 0

    def reset(self) -> None:
        self._pid = 0
        self._base = 0

    def read(self, hwnd: int, addresses: MemoryAddresses) -> MemorySnapshot:
        if addresses.current_sp > 0 and addresses.max_sp > 0:
            return self._read_memory(hwnd, addresses)
        return read_vision_snapshot(hwnd)

    def _read_memory(self, hwnd: int, addresses: MemoryAddresses) -> MemorySnapshot:
        pid = pid_from_hwnd(hwnd)
        if pid <= 0:
            self.reset()
            return MemorySnapshot(error="select a game window")
        if pid != self._pid or self._base <= 0:
            try:
                self._base = module_base_address(pid)
                self._pid = pid
            except OSError as exc:
                self.reset()
                return MemorySnapshot(error=str(exc))
        snap = read_snapshot(pid, self._base, addresses)
        if not snap.ok and "ReadProcessMemory" in snap.error:
            # Base may have shifted after a client restart — refresh once.
            try:
                self._base = module_base_address(pid)
                self._pid = pid
            except OSError as exc:
                self.reset()
                return MemorySnapshot(error=str(exc))
            snap = read_snapshot(pid, self._base, addresses)
        return snap
