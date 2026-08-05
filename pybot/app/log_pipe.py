"""Thread-safe log and status message dispatcher for the tkinter GUI.

Worker threads must never call into Tk directly — doing so (e.g. via
``root.after`` from a non-main thread) blocks the caller until the GUI
thread services the request, which freezes the bot whenever the Tk main
loop stalls (window drag/resize, open menu, any modal loop).

Instead, producers only ``put_nowait`` onto an in-memory queue (truly
non-blocking, thread-safe), and the Tk main thread drains that queue on a
periodic ``after`` poll scheduled from the main thread. A stalled GUI can
therefore only lag the log display, never the hunt runtime.
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
        pipe.set_log_box(text_widget)
        pipe.set_status_widgets(status_label, hint_label)

        # Safe to call from any thread — never blocks:
        pipe.log("Bot started")
        pipe.status("Input: Ready", "Launch the game")
    """

    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._log_box: tk.Text | None = None
        self._status_label: ttk.Label | None = None
        self._hint_label: ttk.Label | None = None
        self._on_append_log: Callable[[str], None] | None = None
        # Optional durable sink for every appended log line (e.g. the session
        # log) so UI-side diagnostics survive the in-memory queue.
        self._on_persist: Callable[[str], None] | None = None
        self._queue: queue.Queue[tuple] = queue.Queue(maxsize=_MAX_QUEUE_ITEMS)
        # __init__ runs on the main thread, so scheduling the drain here
        # keeps every Tk interaction on the main thread.
        self._root.after(_DRAIN_INTERVAL_MS, self._drain)

    # ── Widget registration (call from UI thread after building UI) ──

    def set_log_box(self, widget: tk.Text) -> None:
        """Register the log text widget."""
        self._log_box = widget

    def set_status_widgets(
        self, status_label: ttk.Label, hint_label: ttk.Label
    ) -> None:
        """Register the input status labels."""
        self._status_label = status_label
        self._hint_label = hint_label

    def set_overlay_callback(
        self, callback: Callable[[str], None] | None
    ) -> None:
        """Register a callback fired for every appended log line (e.g. overlay)."""
        self._on_append_log = callback

    def set_persist_callback(
        self, callback: Callable[[str], None] | None
    ) -> None:
        """Register a durable sink fired for every appended log line.

        Unlike the overlay callback (used for transient display), this sink is
        meant for persistence (e.g. writing UI-level lines to the session log)
        so app-side diagnostics are not lost when the in-memory queue dies.

        Note: every line that reaches ``log_pipe.log`` — including runtime
        behavior lines routed through the same pipe — is forwarded here. The
        session log therefore becomes the *complete* session record, while
        ``behavior.log`` (written separately by the runtime) remains the
        runtime-only subset. Overlapping lines are intentionally duplicated
        across the two files.
        """
        self._on_persist = callback

    # ── Thread-safe public API (never blocks) ───────────────────────

    def log(self, message: str) -> None:
        """Queue *message* for the log box (safe from any thread)."""
        self._put(("log", message))

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
                if item[0] == "log":
                    self._do_log(item[1])
                else:
                    self._do_status(item[1], item[2])
                processed += 1
        except queue.Empty:
            pass
        finally:
            self._root.after(_DRAIN_INTERVAL_MS, self._drain)

    def _do_log(self, message: str) -> None:
        if self._log_box is not None:
            self._log_box.configure(state=tk.NORMAL)
            self._log_box.insert(tk.END, message + "\n")
            self._log_box.see(tk.END)
            self._log_box.configure(state=tk.DISABLED)
        if self._on_append_log is not None:
            self._on_append_log(message)
        if self._on_persist is not None:
            self._on_persist(message)

    def _do_status(self, title: str, hint: str) -> None:
        if self._status_label is not None:
            self._status_label.configure(text=title)
        if hint and self._hint_label is not None:
            self._hint_label.configure(text=hint)
