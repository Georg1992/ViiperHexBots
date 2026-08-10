"""SLOW-path diagnostics shared by the discovery and tracking workers.

Emitted only on the rare, throttled SLOW path so the two dump formats cannot
silently diverge. The during-scan sampler records each thread's OS CPU time
alongside its stack, so a stall can be attributed with numbers instead of
guesses: either one bot thread consumed the wall time (GIL/native hog) or
every bot thread was starved while the *game process* burned CPU elsewhere.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np

# Windows-only process/thread CPU probes. Everything here is defensive: on any
# failure the probe returns None and the SLOW log simply omits the field.

def _thread_cpu_seconds(thread_id: int) -> float | None:
    """Kernel+user CPU seconds consumed by *thread_id* (Windows), or None."""
    try:
        import ctypes
        from ctypes import wintypes
    except (ImportError, AttributeError):
        return None
    try:
        THREAD_QUERY_LIMITED_INFORMATION = 0x0800
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenThread(
            THREAD_QUERY_LIMITED_INFORMATION, False, int(thread_id)
        )
        if not handle:
            return None
        try:
            class _FILETIME(ctypes.Structure):
                _fields_ = [
                    ("low", wintypes.DWORD),
                    ("high", wintypes.DWORD),
                ]

            creation, exit_t, kernel, user = (
                _FILETIME(), _FILETIME(), _FILETIME(), _FILETIME()
            )
            if not kernel32.GetThreadTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_t),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                return None

            def _to_seconds(ft: _FILETIME) -> float:
                return (
                    (ft.high << 32) | ft.low
                ) / 1e7

            return _to_seconds(kernel) + _to_seconds(user)
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return None


def game_process_cpu_snapshot(hwnd: int | None) -> tuple[int, float] | None:
    """Return ``(pid, kernel+user CPU seconds)`` of the window's process.

    Used to compare the bot's own CPU against the game client's during a
    SLOW scan: if the game is burning cores while every bot thread is starved,
    the stall is external scheduling pressure, not bot-internal contention.
    """
    if not hwnd:
        return None
    try:
        import ctypes
        from ctypes import wintypes
    except (ImportError, AttributeError):
        return None
    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        pid = wintypes.DWORD()
        if not user32.GetWindowThreadProcessId(
            int(hwnd), ctypes.byref(pid)
        ):
            return None
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value
        )
        if not handle:
            return None
        try:
            class _FILETIME(ctypes.Structure):
                _fields_ = [
                    ("low", wintypes.DWORD),
                    ("high", wintypes.DWORD),
                ]

            creation, exit_t, kernel, user = (
                _FILETIME(), _FILETIME(), _FILETIME(), _FILETIME()
            )
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_t),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                return None

            def _to_seconds(ft: _FILETIME) -> float:
                return (
                    (ft.high << 32) | ft.low
                ) / 1e7

            return int(pid.value), _to_seconds(kernel) + _to_seconds(user)
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return None


def format_thread_dump(max_depth: int = 10) -> str:
    """Snapshot every live thread's Python stack for the SLOW diagnostic path.

    ``sys._current_frames()`` is one cheap O(threads) call — negligible next to
    a multi-second scan. The dump is taken *after* the slow work finished, so
    the calling thread's own stack only shows the logging call site. The useful
    signal is (a) the ``cpuWall`` ratio and (b) whether *another* bot thread's
    stack is stuck inside a long Python/cv2 call — i.e. who actually consumed
    the wall time.
    """
    try:
        frames = sys._current_frames()
    except Exception as exc:  # pragma: no cover - defensive diagnostic
        return f"thread dump unavailable: {exc}"
    by_id = {
        thread.ident: thread.name
        for thread in threading.enumerate()
        if thread.ident is not None
    }
    current_id = threading.get_ident()
    lines = []
    for thread_id, frame in sorted(frames.items()):
        name = by_id.get(thread_id, f"thread-{thread_id}")
        marker = "  <-- this thread" if thread_id == current_id else ""
        lines.append(f"  [{name}]{marker}")
        depth = 0
        current = frame
        while current is not None and depth < max_depth:
            code = current.f_code
            filename = code.co_filename.replace("\\", "/").rsplit("/", 1)[-1]
            lines.append(f"      {filename}:{current.f_lineno} in {code.co_name}")
            current = current.f_back
            depth += 1
    return "\n".join(lines)


def frame_stats(frame: np.ndarray | None) -> str:
    """Cheap stats distinguishing a real game frame from a black/loading frame."""
    if frame is None or frame.size == 0 or frame.ndim < 3:
        return "empty"
    height, width = frame.shape[:2]
    mean = float(frame.mean())
    std = float(frame.std())
    # Sample every 8th pixel for a cheap distinct-color estimate.
    sample = frame[::8, ::8].reshape(-1, frame.shape[2])
    unique = int(np.unique(sample, axis=0).shape[0])
    return (
        f"WxH={width}x{height} mean={mean:.1f} std={std:.1f} "
        f"unique_colors(sample8)={unique}"
    )


def sample_threads_while(
    fn,
    *,
    interval_s: float = 0.5,
    max_samples: int = 20,
):
    """Run *fn* while a daemon thread samples all thread stacks every interval.

    This is the decisive diagnostic for a multi-second wall-time call: it shows
    which threads were *running* while the slow work happened, not just the
    state after it finished. Returns ``(result, samples)`` where each sample is
    ``(elapsed_s, [(thread_name, top_two_frames, cpu_s), ...])`` — ``cpu_s`` is
    that thread's cumulative OS CPU seconds at the sample instant, so callers
    can compute per-thread CPU deltas between samples.

    Only intended for the rare SLOW path; the sampler stops as soon as *fn*
    returns.
    """
    import threading

    samples: list[tuple[float, list[tuple[str, str, float]]]] = []
    stop = threading.Event()
    started = time.monotonic()

    def _collect() -> None:
        while not stop.is_set() and len(samples) < max_samples:
            frames = sys._current_frames()
            by_id = {
                thread.ident: thread.name
                for thread in threading.enumerate()
                if thread.ident is not None
            }
            snapshot: list[tuple[str, str, float]] = []
            for thread_id, frame in sorted(frames.items()):
                name = by_id.get(thread_id, f"thread-{thread_id}")
                code = frame.f_code
                filename = code.co_filename.replace("\\", "/").rsplit("/", 1)[-1]
                top = f"{filename}:{frame.f_lineno} {code.co_name}"
                second = ""
                parent = frame.f_back
                if parent is not None:
                    pcode = parent.f_code
                    pfilename = pcode.co_filename.replace("\\", "/").rsplit("/", 1)[-1]
                    second = f"{pfilename}:{parent.f_lineno} {pcode.co_name}"
                cpu_s = _thread_cpu_seconds(thread_id)
                snapshot.append((name, f"{top} <- {second}", cpu_s or 0.0))
            samples.append((time.monotonic() - started, snapshot))
            stop.wait(interval_s)

    thread = threading.Thread(target=_collect, name="slow-sampler", daemon=True)
    thread.start()
    try:
        result = fn()
    finally:
        stop.set()
        thread.join(timeout=interval_s + 0.5)
    return result, samples


def format_thread_cpu_deltas(
    samples: list[tuple[float, list[tuple[str, str, float]]]],
    min_ms: int = 15,
) -> list[str]:
    """Per-thread CPU attribution from consecutive sampler deltas.

    Each thread's CPU is the sum of non-negative deltas between consecutive
    samples. Only threads that consumed at least *min_ms* of CPU are returned,
    so a hog shows up with a number while idle threads stay silent.
    """
    deltas: dict[str, float] = {}
    previous: dict[str, float] = {}
    for _elapsed, snapshot in samples:
        current = {name: cpu for name, _info, cpu in snapshot}
        for name, cpu in current.items():
            prev = previous.get(name)
            if prev is not None and cpu >= prev:
                deltas[name] = deltas.get(name, 0.0) + (cpu - prev)
        previous = current
    lines = []
    for name, cpu_s in sorted(deltas.items(), key=lambda item: item[1], reverse=True):
        cpu_ms = int(cpu_s * 1000)
        if cpu_ms >= min_ms:
            lines.append(f"  [cpu] {name}: {cpu_ms}ms")
    return lines
