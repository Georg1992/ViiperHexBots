"""Modal editor for the Open Storage key + delay chain (up to 7 steps)."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from pybot.config.schema import MAX_OPEN_STORAGE_STEPS, KeyChainStep
from pybot.runtime.input.scan_codes import keysym_to_key_name


def _pad_chain(steps: list[KeyChainStep]) -> list[KeyChainStep]:
    padded = [
        KeyChainStep(button=s.button, delay_ms=max(0, int(s.delay_ms)))
        for s in steps[:MAX_OPEN_STORAGE_STEPS]
    ]
    while len(padded) < MAX_OPEN_STORAGE_STEPS:
        padded.append(KeyChainStep())
    return padded


def format_storage_chain_summary(steps: list[KeyChainStep]) -> str:
    keys = [s.button.strip() for s in steps if s.button.strip()]
    if not keys:
        return "(none)"
    return " → ".join(keys)


class StorageChainDialog(tk.Toplevel):
    """Seven Key / Delay(ms) columns + Clear, matching the keychain screenshot."""

    def __init__(
        self,
        parent: tk.Misc,
        steps: list[KeyChainStep],
        *,
        on_apply: Callable[[list[KeyChainStep]], None],
    ) -> None:
        super().__init__(parent)
        self.title("Open Storage Keychain")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self._on_apply = on_apply
        self._steps = _pad_chain(steps)
        self._key_vars: list[tk.StringVar] = []
        self._delay_vars: list[tk.StringVar] = []
        self._key_entries: list[ttk.Entry] = []

        body = ttk.Frame(self, padding=10)
        body.pack(fill=tk.BOTH, expand=True)

        row = ttk.Frame(body)
        row.pack(fill=tk.X)

        for i in range(MAX_OPEN_STORAGE_STEPS):
            col = ttk.Frame(row, padding=(4, 0))
            col.pack(side=tk.LEFT, padx=2)

            ttk.Label(col, text="Keys:").pack(anchor="w")
            key_var = tk.StringVar(value=self._steps[i].button or "")
            self._key_vars.append(key_var)
            key_entry = ttk.Entry(col, textvariable=key_var, width=8, justify="center")
            key_entry.pack()
            key_entry.bind("<KeyPress>", self._on_key_capture)
            self._key_entries.append(key_entry)

            ttk.Label(col, text="↓", font=("Segoe UI", 8)).pack()

            ttk.Label(col, text="Delay(ms):").pack(anchor="w")
            delay_var = tk.StringVar(value=str(self._steps[i].delay_ms))
            self._delay_vars.append(delay_var)
            delay_entry = ttk.Entry(col, textvariable=delay_var, width=8, justify="center")
            delay_entry.pack()

        clear_btn = ttk.Button(body, text="Clear", command=self._on_clear)
        clear_btn.pack(fill=tk.X, pady=(12, 0))

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Escape>", lambda _e: self._on_close())

        self.update_idletasks()
        try:
            px = parent.winfo_rootx() + 40
            py = parent.winfo_rooty() + 80
            self.geometry(f"+{px}+{py}")
        except tk.TclError:
            pass
        if self._key_entries:
            self._key_entries[0].focus_set()

    def _on_key_capture(self, event: tk.Event) -> str:
        widget = event.widget
        if event.keysym in ("BackSpace", "Delete"):
            widget.delete(0, tk.END)
            return "break"
        name = keysym_to_key_name(event.keysym)
        if not name:
            return "break"
        widget.delete(0, tk.END)
        widget.insert(0, name)
        return "break"

    def _collect(self) -> list[KeyChainStep]:
        collected: list[KeyChainStep] = []
        for key_var, delay_var in zip(self._key_vars, self._delay_vars):
            button = key_var.get().strip()
            raw = delay_var.get().strip()
            delay = int(raw) if raw.isdigit() else 0
            collected.append(KeyChainStep(button=button, delay_ms=max(0, delay)))
        # Drop trailing empty slots for storage; keep leading empties as gaps? Plan:
        # trailing empty ignored — store only assigned keys in order, but UI pads to 7.
        # Persist non-empty steps only (same as skill timers).
        return [s for s in collected if s.button]

    def _on_clear(self) -> None:
        for key_var, delay_var in zip(self._key_vars, self._delay_vars):
            key_var.set("")
            delay_var.set("0")

    def _on_close(self) -> None:
        self._on_apply(self._collect())
        self.grab_release()
        self.destroy()
