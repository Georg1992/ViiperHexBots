"""Thread-safe log and status message dispatcher for the tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk


class LogPipe:
    """Thread-safe log/status dispatcher that uses root.after() for GUI updates.

    Usage::

        pipe = LogPipe(root)
        pipe.set_log_box(text_widget)
        pipe.set_status_widgets(status_label, hint_label)

        # Safe to call from any thread:
        pipe.log("Bot started")
        pipe.status("Input: Ready", "Launch the game")
    """

    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._log_box: tk.Text | None = None
        self._status_label: ttk.Label | None = None
        self._hint_label: ttk.Label | None = None
        self._on_append_log: Callable[[str], None] | None = None

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

    # ── Thread-safe public API ──────────────────────────────────────

    def log(self, message: str) -> None:
        """Append *message* to the log box (safe from any thread)."""
        self._root.after(0, self._do_log, message)

    def status(self, title: str, hint: str = "") -> None:
        """Update the input status labels (safe from any thread)."""
        self._root.after(0, self._do_status, title, hint)

    # ── Internal dispatch (runs on main thread via root.after) ──────

    def _do_log(self, message: str) -> None:
        if self._log_box is not None:
            self._log_box.configure(state=tk.NORMAL)
            self._log_box.insert(tk.END, message + "\n")
            self._log_box.see(tk.END)
            self._log_box.configure(state=tk.DISABLED)
        if self._on_append_log is not None:
            self._on_append_log(message)

    def _do_status(self, title: str, hint: str) -> None:
        if self._status_label is not None:
            self._status_label.configure(text=title)
        if hint and self._hint_label is not None:
            self._hint_label.configure(text=hint)
