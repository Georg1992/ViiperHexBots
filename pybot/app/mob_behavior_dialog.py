"""Modal editor for per-mob custom hunt behavior."""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from pybot.config.schema import MobCustomSettings
from pybot.runtime.input.scan_codes import keysym_to_key_name


class MobBehaviorDialog(tk.Toplevel):
    """Edit kiting and self-cast skill settings for one mob."""

    def __init__(
        self,
        parent: tk.Misc,
        mob_name: str,
        settings: MobCustomSettings,
        *,
        on_apply: Callable[[MobCustomSettings], None],
    ) -> None:
        super().__init__(parent)
        self.title(f"Custom behavior — {mob_name}")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self._on_apply = on_apply

        body = ttk.Frame(self, padding=12)
        body.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            body,
            text="Cast the debuff once per target, heal safely before attacks, and kite during the attack delay.",
            wraplength=360,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        ttk.Label(body, text="Kiting tick (s):").grid(row=1, column=0, sticky="w")
        self._kiting_tick = ttk.Entry(body, width=8)
        self._kiting_tick.insert(0, self._format_number(settings.kiting_tick_s))
        self._kiting_tick.grid(row=1, column=1, sticky="w", padx=(8, 0))

        ttk.Separator(body, orient=tk.HORIZONTAL).grid(
            row=2, column=0, columnspan=3, sticky="ew", pady=10
        )

        ttk.Label(body, text="Skill", font=("Segoe UI", 9, "bold")).grid(
            row=3, column=0, sticky="w"
        )
        ttk.Label(body, text="Button", font=("Segoe UI", 9, "bold")).grid(
            row=3, column=1, sticky="w", padx=(8, 0)
        )
        ttk.Label(body, text="Delay (s)", font=("Segoe UI", 9, "bold")).grid(
            row=3, column=2, sticky="w", padx=(8, 0)
        )

        self._key_entries: list[ttk.Entry] = []
        self._delay_entries: list[ttk.Entry] = []
        rows = [
            ("Debuff", settings.debuff_button, None),
            ("Heal", settings.heal_button, None),
            ("Buff 1", settings.buff1_button, settings.buff1_delay_s),
            ("Buff 2", settings.buff2_button, settings.buff2_delay_s),
            ("Buff 3", settings.buff3_button, settings.buff3_delay_s),
        ]
        for row, (label, button, delay) in enumerate(rows, start=4):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky="w", pady=2)
            key_entry = ttk.Entry(body, width=10)
            key_entry.insert(0, button)
            key_entry.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=2)
            key_entry.bind("<KeyPress>", self._on_key_capture)
            self._key_entries.append(key_entry)

            if delay is not None:
                delay_entry = ttk.Entry(body, width=8)
                delay_entry.insert(0, str(delay))
                delay_entry.grid(row=row, column=2, sticky="w", padx=(8, 0), pady=2)
                self._delay_entries.append(delay_entry)
            else:
                ttk.Label(body, text="—").grid(
                    row=row, column=2, sticky="w", padx=(8, 0), pady=2
                )

        buttons = ttk.Frame(body)
        buttons.grid(row=9, column=0, columnspan=3, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="Cancel", command=self._cancel).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Save", command=self._save).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda _event: self._cancel())
        self.update_idletasks()
        try:
            self.geometry(f"+{parent.winfo_rootx() + 60}+{parent.winfo_rooty() + 100}")
        except tk.TclError:
            pass
        self._key_entries[0].focus_set()

    @staticmethod
    def _format_number(value: float) -> str:
        return f"{value:g}"

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

    def _save(self) -> None:
        try:
            kiting_tick = float(self._kiting_tick.get().strip() or "0")
            delays = [int(entry.get().strip() or "0") for entry in self._delay_entries]
        except ValueError:
            messagebox.showerror(
                "Invalid custom behavior",
                "Kiting tick and buff delays must be valid non-negative numbers.",
                parent=self,
            )
            return
        if not math.isfinite(kiting_tick) or kiting_tick < 0 or any(delay < 0 for delay in delays):
            messagebox.showerror(
                "Invalid custom behavior",
                "Kiting tick and buff delays must be valid non-negative numbers.",
                parent=self,
            )
            return
        result = MobCustomSettings(
            kiting_tick_s=kiting_tick,
            debuff_button=self._key_entries[0].get().strip(),
            heal_button=self._key_entries[1].get().strip(),
            buff1_button=self._key_entries[2].get().strip(),
            buff1_delay_s=delays[0],
            buff2_button=self._key_entries[3].get().strip(),
            buff2_delay_s=delays[1],
            buff3_button=self._key_entries[4].get().strip(),
            buff3_delay_s=delays[2],
        )

        self._on_apply(result)
        self.destroy()

    def _cancel(self) -> None:
        self.destroy()
