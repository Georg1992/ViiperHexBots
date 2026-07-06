"""Memory reading via ReadProcessMemory — Reads SP, weight, and current location from the game process using
addresses defined in the client profile JSON.
"""

from __future__ import annotations

import ctypes
import json
import threading
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

kernel32 = ctypes.windll.kernel32

# SIZE_T was removed from ctypes.wintypes in Python 3.14
if not hasattr(wintypes, "SIZE_T"):
    wintypes.SIZE_T = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
TH32CS_SNAPPROCESS = 0x0002


@dataclass
class GameMemoryState:
    max_sp: int = 0
    current_sp: int = 0
    current_weight: int = 0
    total_weight: int = 0
    current_location: int = 0


@dataclass
class ClientMemoryLayout:
    max_sp_address: int = 0
    current_sp_address: int = 0
    current_weight_address: int = 0
    total_weight_address: int = 0
    current_location_address: int = 0

    @property
    def active(self) -> bool:
        return bool(self.current_location_address)


def load_memory_layout(client_profile: str, project_root: Path) -> ClientMemoryLayout:
    """Read memory addresses from the client profile JSON."""
    profile_path = project_root / "clients" / f"{client_profile}.json"
    if not profile_path.is_file():
        return ClientMemoryLayout()
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ClientMemoryLayout()

    memory = data.get("memory")
    if not isinstance(memory, dict):
        return ClientMemoryLayout()

    def _addr(key: str) -> int:
        raw = memory.get(key, 0)
        if isinstance(raw, str):
            return int(raw, 16) if raw.startswith("0x") else int(raw)
        return int(raw) if raw else 0

    return ClientMemoryLayout(
        max_sp_address=_addr("maxSpAddress"),
        current_sp_address=_addr("currentSpAddress"),
        current_weight_address=_addr("currentWeightAddress"),
        total_weight_address=_addr("totalWeightAddress"),
        current_location_address=_addr("currentLocationAddress"),
    )


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


class MemoryReader:
    """Read game memory via ReadProcessMemory in a thread-safe manner."""

    def __init__(
        self,
        process_name: str,
        layout: ClientMemoryLayout,
        *,
        on_state: Callable[[GameMemoryState], None] | None = None,
    ) -> None:
        self._process_name = process_name
        self._layout = layout
        self._on_state = on_state
        self._lock = threading.Lock()
        self._state = GameMemoryState()
        self._pid = 0

    @property
    def active(self) -> bool:
        return self._layout.active

    def update(self) -> GameMemoryState:
        """Read all memory addresses and return the new state."""
        if not self._layout.active:
            return self._state

        pid = self._resolve_pid()
        if not pid:
            return self._state

        with self._lock:
            state = GameMemoryState(
                max_sp=self._read_uint(pid, self._layout.max_sp_address),
                current_sp=self._read_uint(pid, self._layout.current_sp_address),
                current_weight=self._read_uint(pid, self._layout.current_weight_address),
                total_weight=self._read_uint(pid, self._layout.total_weight_address),
                current_location=self._read_uint(pid, self._layout.current_location_address),
            )
            self._state = state

        if self._on_state:
            self._on_state(state)

        return state

    @property
    def state(self) -> GameMemoryState:
        with self._lock:
            return self._state

    def _resolve_pid(self) -> int:
        # Check cached PID first
        if self._pid:
            handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, self._pid)
            if handle:
                kernel32.CloseHandle(handle)
                return self._pid
            self._pid = 0

        # Enumerate processes via Toolhelp32Snapshot
        snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snapshot == -1:
            return 0

        try:
            pe = PROCESSENTRY32W()
            pe.dwSize = ctypes.sizeof(pe)
            if not kernel32.Process32FirstW(snapshot, ctypes.byref(pe)):
                return 0

            proc_name_lower = self._process_name.lower()
            while True:
                exe = (pe.szExeFile or "").lower()
                # Match exact executable name
                if exe == proc_name_lower:
                    self._pid = pe.th32ProcessID
                    return self._pid
                if not kernel32.Process32NextW(snapshot, ctypes.byref(pe)):
                    break
        finally:
            kernel32.CloseHandle(snapshot)

        return 0

    @staticmethod
    def _read_uint(pid: int, address: int) -> int:
        if not address:
            return 0
        handle = kernel32.OpenProcess(PROCESS_VM_READ, False, pid)
        if not handle:
            return 0
        try:
            buf = ctypes.create_string_buffer(4)
            read = wintypes.SIZE_T()
            success = kernel32.ReadProcessMemory(
                handle, ctypes.c_void_p(address), buf, 4, ctypes.byref(read),
            )
            if success and read.value == 4:
                return int.from_bytes(buf.raw[:4], "little")
            return 0
        finally:
            kernel32.CloseHandle(handle)
