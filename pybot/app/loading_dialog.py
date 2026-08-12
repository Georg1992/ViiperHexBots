"""Reusable modal loading UI for long-running Tk operations."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class LoadingDialog:
    """Show a centered, modal indeterminate-progress loading dialog.

    The dialog deliberately owns no worker thread and never calls Tk from a
    worker. Callers start their work separately, update the message on the Tk
    thread, and call :meth:`close` when the work completes.

    When ``parent`` is omitted, a standalone ``Tk`` root is created. This
    supports the pre-main-window descriptor splash while keeping the visual
    loading treatment shared with modal operations in the main window.
    """

    def __init__(
        self,
        parent: tk.Misc | None = None,
        *,
        title: str,
        heading: str,
        message: str,
        width: int = 420,
    ) -> None:
        self._owns_root = parent is None
        self.window = tk.Tk() if parent is None else tk.Toplevel(parent)
        self.window.title(title)
        self.window.resizable(False, False)
        if parent is not None:
            self.window.transient(parent)

        frame = ttk.Frame(self.window, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            frame,
            text=heading,
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")

        self._progress = ttk.Progressbar(
            frame,
            mode="indeterminate",
            length=width,
        )
        self._progress.pack(fill=tk.X, pady=(12, 0))
        self._progress.start(12)

        self._status = tk.StringVar(value=message)
        ttk.Label(
            frame,
            textvariable=self._status,
            wraplength=width,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(12, 0))

        self.window.protocol("WM_DELETE_WINDOW", self._ignore_close)
        self.window.update_idletasks()
        if parent is None:
            screen_w = self.window.winfo_screenwidth()
            screen_h = self.window.winfo_screenheight()
            x = (screen_w - self.window.winfo_reqwidth()) // 2
            y = (screen_h - self.window.winfo_reqheight()) // 2
            self.window.geometry(f"+{x}+{y}")
        else:
            self._center_over_parent(parent)
            self.window.grab_set()
        self.window.lift()

    def _center_over_parent(self, parent: tk.Misc) -> None:
        """Center the dialog over its parent after its requested size is known."""
        try:
            parent.update_idletasks()
            x = parent.winfo_rootx() + (
                parent.winfo_width() - self.window.winfo_reqwidth()
            ) // 2
            y = parent.winfo_rooty() + (
                parent.winfo_height() - self.window.winfo_reqheight()
            ) // 2
            self.window.geometry(f"+{max(0, x)}+{max(0, y)}")
        except tk.TclError:
            # The owner may be closing while an operation completion is queued.
            pass

    def _ignore_close(self) -> None:
        """Keep the modal operation visible until its worker completes."""

    def set_message(self, message: str) -> None:
        """Update the status text; must be called on the Tk thread."""
        try:
            self._status.set(message)
            self.window.update_idletasks()
        except tk.TclError:
            pass

    def stop(self) -> None:
        """Stop the progress animation while leaving the dialog visible."""
        try:
            self._progress.stop()
        except tk.TclError:
            pass

    def close(self) -> None:
        """Stop the animation and release/destroy the dialog."""
        try:
            self.stop()
            if not self._owns_root:
                self.window.grab_release()
            self.window.destroy()
        except tk.TclError:
            pass
