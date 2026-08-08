"""Thread-safe status dispatcher and durable log sink for the tkinter GUI.

Worker threads must never call into Tk directly. Status updates are queued
for the Tk thread, while log messages go straight to the durable session
sink. This keeps file logging independent from GUI responsiveness now that
logs are intentionally not displayed in the window or hunt overlay.
"""

from __future__ import annotations

import queue
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

_DRAIN_INTERVAL_MS = 50
_MAX_ITEMS_PER_DRAIN = 30
_MAX_QUEUE_ITEMS = 2000


class LogPipe:
    """Thread-safe log/status dispatcher backed by a queue drained on the UI thread.

    Usage::

        pipe = LogPipe(root)              # starts the drain loop (main thread)
        pipe.set_status_widgets(status_label, hint_label)
        pipe.set_persist_callback(session_writer)

        # Safe to call from any thread — never blocks:
        pipe.log("Bot started")
        pipe.status("Input: Ready", "Launch the game")
    """

    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._status_label: ttk.Label | None = None
        self._hint_label: ttk.Label | None = None
        # Optional durable sink for every log call (e.g. the session log).
        # Log calls invoke this directly, so persistence does not depend on
        # the Tk event loop draining a display queue.
        self._on_persist: Callable[[str], None] | None = None
        self._queue: queue.Queue[tuple] = queue.Queue(maxsize=_MAX_QUEUE_ITEMS)
        # __init__ runs on the main thread, so scheduling the status drain here
        # keeps every Tk interaction on the main thread.
        self._root.after(_DRAIN_INTERVAL_MS, self._drain)

    # ── Widget registration (call from UI thread after building UI) ──

    def set_status_widgets(
        self, status_label: ttk.Label, hint_label: ttk.Label
    ) -> None:
        """Register the input status labels."""
        self._status_label = status_label
        self._hint_label = hint_label

    def set_persist_callback(
        self, callback: Callable[[str], None] | None
    ) -> None:
        """Register the durable sink for every subsequent log call.

        The callback is invoked directly from the calling thread and must be
        non-blocking/thread-safe; ``AppSessionLog.write_system`` satisfies
        that contract. Runtime behavior logs also remain in ``behavior.log``
        through ``HuntLogger``; the duplicated app/session record is
        intentional.
        """
        self._on_persist = callback

    # ── Thread-safe public API (never blocks) ───────────────────────

    def log(self, message: str) -> None:
        """Persist *message* without touching Tk (safe from any thread)."""
        if self._on_persist is not None:
            self._on_persist(message)

    def status(self, title: str, hint: str = "") -> None:
        """Queue an input status update (safe from any thread)."""
        self._put(("status", title, hint))

    def _put(self, item: tuple) -> None:
        """Keep producer calls non-blocking and bound memory when UI stalls."""
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                pass

    # ── Internal dispatch (runs on main thread via root.after) ──────

    def _drain(self) -> None:
        processed = 0
        try:
            while processed < _MAX_ITEMS_PER_DRAIN:
                item = self._queue.get_nowait()
                self._do_status(item[1], item[2])
                processed += 1
        except queue.Empty:
            pass
        finally:
            self._root.after(_DRAIN_INTERVAL_MS, self._drain)

    def _do_status(self, title: str, hint: str) -> None:
        if self._status_label is not None:
            self._status_label.configure(text=title)
        if hint and self._hint_label is not None:
            self._hint_label.configure(text=hint)
