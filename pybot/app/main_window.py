"""ViiperHexBots main window (tkinter) — UI building and callback wiring only.

Lifecycle logic             → :mod:`pybot.app.bot_lifecycle`
Hotkey registration/polling → :mod:`pybot.app.hotkey_manager`
Thread-safe log dispatch    → :mod:`pybot.app.log_pipe`
"""

from __future__ import annotations

import ctypes
import threading
import tkinter as tk
from ctypes import wintypes
from tkinter import messagebox, ttk

from pybot.app.bot_lifecycle import BotLifecycleManager, BotState
from pybot.app.bot_controller import DEFAULT_STOP_JOIN_TIMEOUT_S
from pybot.app.config_store import AppConfig, list_client_profiles
from pybot.config.clients import memory_reading_enabled
from pybot.app.hotkey_manager import HotkeyManager
from pybot.app.log_pipe import LogPipe
from pybot.app.overlay import Win32HuntOverlay
from pybot.mobs.catalog import load_mob_catalog
from pybot.app.session_log import AppSessionLog
from pybot.app.startup_splash import preload_mob_descriptors
from pybot.app.viiper_manager import ViiperManager
from pybot.app.win32_util import (
    enum_game_windows,
    restore_and_activate,
    window_exists,
)

user32 = ctypes.windll.user32


class MainWindow:
    """Build the tkinter UI and wire lifecycle/log/hotkey managers together."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Hex Bot")
        self.root.geometry("920x780")
        self.root.minsize(880, 720)

        # ── Data layer ──────────────────────────────────────────────
        self.config = AppConfig().load()
        self.mob_catalog = load_mob_catalog(ensure_assets=False)
        self._check_mob_catalog()
        self.session = AppSessionLog()
        self._hunt_overlay = Win32HuntOverlay()

        # ── Managers (created before UI so callbacks are ready) ─────
        self.log_pipe = LogPipe(self.root)
        self.viiper = ViiperManager(
            on_log=self.log_pipe.log,
            on_status=self.log_pipe.status,
        )
        self.lifecycle = BotLifecycleManager(
            root=self.root,
            config=self.config,
            mob_catalog=self.mob_catalog,
            session=self.session,
            viiper=self.viiper,
            hunt_overlay=self._hunt_overlay,
            on_state_change=self._on_bot_state_changed,
            on_log=self.log_pipe.log,
            on_input_ready=self._enable_after_viiper,
            on_exit_requested=self.on_exit,
        )
        self.hotkey_manager = HotkeyManager(
            root=self.root,
            on_hotkey=self.toggle_bot,
        )

        # ── UI state ────────────────────────────────────────────────

        self.window_entries: list = []
        self.mob_var = tk.IntVar(
            value=min(max(1, self.config.selected_monster), len(self.mob_catalog))
        )

        # Build UI (widgets created here, references shared to managers)
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        # Wire log pipe UI references after widgets exist
        self.log_pipe.set_log_box(self.log_box)
        self.log_pipe.set_status_widgets(self.input_status, self.input_hint)
        self.log_pipe.set_overlay_callback(self._maybe_pipe_to_overlay)

        # Async VIIPER init (descriptors were prepared on the splash before this window)
        self.log_pipe.log("ViiperHexBots started (Python)")
        self.log_pipe.log("Starting VIIPER before game launch...")
        threading.Thread(target=self.lifecycle.init_viiper, daemon=True).start()

    # ── Pre-flight ──────────────────────────────────────────────────

    def _check_mob_catalog(self) -> None:
        if not self.mob_catalog:
            messagebox.showerror(
                "ViiperHexBots",
                "No mob descriptors found.\n\n"
                "Create one with:\n.\\scripts\\build-mob-descriptor.ps1 -Mob horn",
            )
            raise SystemExit(1)

    # ══════════════════════════════════════════════════════════════════
    #  UI BUILDING
    # ══════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="ViiperHex Bot", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8)
        )

        # ── Game Window ───────────────────────────────────────────
        window_frame = ttk.LabelFrame(main, text="Game Window", padding=8)
        window_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=(0, 8))
        ttk.Label(window_frame, text="Select game window:").grid(
            row=0, column=0, sticky="w"
        )
        self.window_combo = ttk.Combobox(window_frame, state="readonly", width=62)
        self.window_combo.grid(row=1, column=0, sticky="ew", pady=4)
        self.window_combo.bind("<<ComboboxSelected>>", self.on_window_selected)
        ttk.Button(window_frame, text="Refresh", command=self.refresh_windows).grid(
            row=1, column=1, padx=(8, 0)
        )
        self.window_info = ttk.Label(window_frame, text="No window selected")
        self.window_info.grid(row=2, column=0, columnspan=2, sticky="w")
        window_frame.columnconfigure(0, weight=1)

        # ── Status & Input (compact side panel) ─────────────────────
        status_input_frame = ttk.LabelFrame(main, text="Status & Input", padding=6)
        status_input_frame.grid(row=1, column=2, sticky="nsew")
        self.bot_status = ttk.Label(status_input_frame, text="Status: Off")
        self.bot_status.pack(anchor="w")
        self.status_indicator = tk.Label(
            status_input_frame, text="  OFF  ", bg="#c62828", fg="white",
            font=("Segoe UI", 9, "bold"), width=10,
        )
        self.status_indicator.pack(anchor="w", pady=(4, 6))
        self.input_status = ttk.Label(status_input_frame, text="Input: Starting...")
        self.input_status.pack(anchor="w")
        self.input_hint = ttk.Label(
            status_input_frame,
            text="Launch the game after VIIPER is ready",
        )
        self.input_hint.pack(anchor="w", pady=(2, 0))
        profile_row = ttk.Frame(status_input_frame)
        profile_row.pack(anchor="w", fill="x", pady=(8, 0))
        ttk.Label(profile_row, text="Client Profile:").pack(side=tk.LEFT)
        self.client_combo = ttk.Combobox(
            profile_row,
            values=list_client_profiles(),
            state="readonly",
            width=14,
        )
        self.client_combo.set(self.config.client_profile)
        self.client_combo.pack(side=tk.LEFT, padx=(6, 0))
        self.client_combo.bind("<<ComboboxSelected>>", self.on_client_changed)

        # ── Setup ──────────────────────────────────────────────────
        setup_frame = ttk.LabelFrame(main, text="Setup", padding=8)
        setup_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 8), pady=(8, 0))
        ttk.Label(setup_frame, text="Descriptor Mob:").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        mob_row = 1
        for index, mob in enumerate(self.mob_catalog, start=1):
            ttk.Radiobutton(
                setup_frame,
                text=mob.display_name,
                variable=self.mob_var,
                value=index,
            ).grid(row=mob_row, column=0, columnspan=2, sticky="w")
            mob_row += 1
        # ── Keybindings ─────────────────────────────────────────────
        keys_frame = ttk.LabelFrame(main, text="Keybindings", padding=8)
        keys_frame.grid(row=2, column=1, sticky="nsew", pady=(8, 0))
        self.skill_button = self._labeled_entry(
            keys_frame, "Attack Skill:", self.config.skill_button, 0
        )
        self.skill_delay = self._labeled_entry(
            keys_frame, "Attack delay (ms):", str(self.config.skill_delay), 1
        )
        self.teleport_button = self._labeled_entry(
            keys_frame, "Teleport:", self.config.teleport_button, 2
        )
        self.save_point_button = self._labeled_entry(
            keys_frame, "Save Point:", self.config.save_point_button, 3
        )
        self.open_storage_button = self._labeled_entry(
            keys_frame, "Open Storage:", self.config.open_storage_button, 4
        )
        self.skill_timer_button = self._labeled_entry(
            keys_frame, "Skill Timer:", self.config.skill_timer_button, 5
        )
        self.skill_timer_interval = self._labeled_entry(
            keys_frame, "Skill Timer (s):", str(self.config.skill_timer_interval), 6
        )
        self.sp_button = self._labeled_entry(
            keys_frame, "SP Item:", self.config.sp_button, 7
        )

        # ── Hunt Settings ───────────────────────────────────────────
        hunt_frame = ttk.LabelFrame(main, text="Hunt Settings", padding=8)
        hunt_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Label(hunt_frame, text="Search Range (9-16 cells):").grid(
            row=0, column=0, sticky="w"
        )
        self.search_range = tk.IntVar(value=self.config.search_range)
        self.search_scale = ttk.Scale(
            hunt_frame,
            from_=9,
            to=16,
            orient=tk.HORIZONTAL,
            variable=self.search_range,
            command=self._update_search_label,
        )
        self.search_scale.grid(row=0, column=1, sticky="ew", padx=8)
        self.search_label = ttk.Label(hunt_frame, text=str(self.config.search_range))
        self.search_label.grid(row=0, column=2)
        ttk.Label(hunt_frame, text="Hunt mode:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.hunt_mode_var = tk.StringVar(value=self.config.hunt_mode)
        self.hunt_mode_combo = ttk.Combobox(
            hunt_frame,
            textvariable=self.hunt_mode_var,
            values=("teleport", "walk"),
            state="readonly",
            width=12,
        )
        self.hunt_mode_combo.grid(row=1, column=1, sticky="w", padx=8, pady=(8, 0))
        ttk.Label(hunt_frame, text="Items To Kafra when weight is:").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        self.weight_modifier = tk.IntVar(value=self.config.weight_modifier)
        self.weight_scale = ttk.Scale(
            hunt_frame,
            from_=49,
            to=90,
            orient=tk.HORIZONTAL,
            variable=self.weight_modifier,
            command=self._update_weight_label,
        )
        self.weight_scale.grid(row=2, column=1, sticky="ew", padx=8, pady=(8, 0))
        self.weight_label = ttk.Label(hunt_frame, text=self._weight_text())
        self.weight_label.grid(row=2, column=2, pady=(8, 0))
        self.fly_wings_var = tk.BooleanVar(value=self.config.take_fly_wings)
        ttk.Checkbutton(
            hunt_frame, text="Take Fly Wings", variable=self.fly_wings_var
        ).grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.fly_wings_amount = ttk.Entry(hunt_frame, width=6)
        self.fly_wings_amount.insert(0, str(self.config.fly_wings_amount))
        self.fly_wings_amount.grid(row=3, column=1, sticky="w", pady=(8, 0))
        self.captcha_var = tk.BooleanVar(value=self.config.detect_captcha)
        ttk.Checkbutton(
            hunt_frame, text="Detect Captcha", variable=self.captcha_var
        ).grid(row=3, column=2, sticky="w", pady=(8, 0))

        hunt_frame.columnconfigure(1, weight=1)

        # ── Warper ──────────────────────────────────────────────────
        warper_frame = ttk.LabelFrame(main, text="Warper Coordinates", padding=8)
        warper_frame.grid(row=2, column=2, sticky="nsew", pady=(8, 0))
        ttk.Button(
            warper_frame, text="Set Position", command=self.on_set_warper
        ).pack(anchor="w")
        ttk.Button(
            warper_frame, text="Reset", command=self.on_reset_warper
        ).pack(anchor="w", pady=(4, 0))
        self.warper_label = ttk.Label(
            warper_frame,
            text=self._warper_text(),
        )
        self.warper_label.pack(anchor="w", pady=(8, 0))
        ttk.Label(warper_frame, text="Time on location:").pack(
            anchor="w", pady=(8, 0)
        )
        self.time_on_location = tk.IntVar(value=self.config.time_on_location)
        ttk.Scale(
            warper_frame,
            from_=20,
            to=240,
            orient=tk.HORIZONTAL,
            variable=self.time_on_location,
        ).pack(fill=tk.X)

        # ── Log (full width, expands with window) ───────────────────
        log_frame = ttk.LabelFrame(main, text="Log", padding=8)
        log_frame.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        log_body = ttk.Frame(log_frame)
        log_body.pack(fill=tk.BOTH, expand=True)
        log_scroll = ttk.Scrollbar(log_body, orient=tk.VERTICAL)
        self.log_box = tk.Text(
            log_body,
            height=10,
            state=tk.DISABLED,
            wrap=tk.WORD,
            yscrollcommand=log_scroll.set,
        )
        log_scroll.config(command=self.log_box.yview)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.overlay_var = tk.BooleanVar(value=self.config.hunt_log_overlay)
        ttk.Checkbutton(
            log_frame, text="Hunt log overlay on game", variable=self.overlay_var
        ).pack(anchor="w", pady=(6, 0))

        # ── Controls (pinned below log, never clipped) ──────────────
        controls = ttk.Frame(main)
        controls.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        ttk.Label(controls, text="Press F12 to quickly toggle bot").pack()
        button_row = ttk.Frame(controls)
        button_row.pack(pady=8)
        ttk.Button(button_row, text="Exit", command=self.on_exit).pack(
            side=tk.LEFT, padx=6
        )
        self.bot_button = ttk.Button(
            button_row,
            text="Start Bot",
            command=self.toggle_bot,
            state=tk.DISABLED,
        )
        self.bot_button.pack(side=tk.LEFT, padx=6)
        self.continue_button = ttk.Button(
            button_row,
            text="Continue",
            command=self.resume_bot,
            state=tk.DISABLED,
        )
        self.continue_button.pack(side=tk.LEFT, padx=6)

        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.columnconfigure(2, weight=1)
        main.rowconfigure(4, weight=1)
        self._sync_memory_reading_from_profile()
        self._update_search_label()

    def _labeled_entry(
        self, parent, label: str, value: str, row: int
    ) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, width=10)
        entry.insert(0, value)
        entry.grid(row=row, column=1, sticky="w", pady=2)
        return entry

    # ══════════════════════════════════════════════════════════════════
    #  UI CALLBACKS (widget value helpers)
    # ══════════════════════════════════════════════════════════════════

    def _warper_text(self) -> str:
        if self.config.warper_coords_set:
            return f"X: {self.config.warper_x} Y: {self.config.warper_y}"
        return "Not set"

    def _weight_text(self) -> str:
        value = int(self.weight_modifier.get())
        return "Off" if value == 49 else str(value)

    def _update_search_label(self, *_args) -> None:
        cells = int(float(self.search_range.get()))
        px = cells * 64
        self.search_label.configure(text=f"{cells} ({px}px)")
        self.lifecycle.set_search_range_cells(cells)

    def _update_weight_label(self, *_args) -> None:
        self.weight_label.configure(text=self._weight_text())

    def refresh_windows(self) -> None:
        self.window_entries = enum_game_windows(
            exclude_hwnd=self.root.winfo_id()
        )
        labels = [entry.display_text for entry in self.window_entries]
        self.window_combo["values"] = labels
        selected = ""
        if self.config.window_id:
            for entry in self.window_entries:
                if entry.hwnd == self.config.window_id:
                    selected = entry.display_text
                    break
        if (
            not selected
            and self.config.last_session_title
            and self.config.last_session_process
        ):
            for entry in self.window_entries:
                if (
                    entry.title == self.config.last_session_title
                    and entry.process == self.config.last_session_process
                ):
                    selected = entry.display_text
                    break
        if selected:
            self.window_combo.set(selected)
            self.on_window_selected()
        elif labels:
            self.window_combo.current(0)
            self.on_window_selected()

    def on_window_selected(self, *_event) -> None:
        label = self.window_combo.get()
        entry = next(
            (item for item in self.window_entries if item.display_text == label),
            None,
        )
        if entry is None:
            return
        self.config.window_id = entry.hwnd
        self.config.window_title = entry.title
        self.config.window_process = entry.process
        self.config.last_session_title = entry.title
        self.config.last_session_process = entry.process
        self.window_info.configure(text=entry.display_text)

    def on_client_changed(self, *_event) -> None:
        self.config.client_profile = self.client_combo.get()
        self._sync_memory_reading_from_profile()
        memory = "on" if self.config.use_memory_reading else "off"
        self.log_pipe.log(
            f"Client profile: {self.config.client_profile} (memory reading {memory})"
        )

    def _sync_memory_reading_from_profile(self) -> None:
        """Memory reading follows the profile: Generic off, server profiles on."""
        self.config.use_memory_reading = memory_reading_enabled(self.client_combo.get())

    def on_set_warper(self) -> None:
        """Capture current cursor position as warper coords."""
        if not self.config.window_id or not window_exists(self.config.window_id):
            messagebox.showwarning("Warper", "Please select a game window first.")
            return
        restore_and_activate(self.config.window_id)
        pos = wintypes.POINT()
        if not user32.GetCursorPos(ctypes.byref(pos)):
            messagebox.showerror("Error", "Could not read cursor position.")
            return
        self.config.warper_x = str(pos.x)
        self.config.warper_y = str(pos.y)
        self.config.warper_location = 0
        self.warper_label.configure(text=self._warper_text())
        self.log_pipe.log(f"Warper position set to X={pos.x} Y={pos.y}")

    def on_reset_warper(self) -> None:
        self.config.warper_x = ""
        self.config.warper_y = ""
        self.config.warper_location = 0
        self.warper_label.configure(text=self._warper_text())

    def _sync_config_from_ui(self) -> None:
        """Read all UI widget values into self.config."""
        self.config.client_profile = self.client_combo.get()
        self._sync_memory_reading_from_profile()
        self.config.selected_monster = self.mob_var.get()
        self.config.hunt_mode = self.hunt_mode_var.get()
        self.config.search_range = int(float(self.search_range.get()))
        self.config.weight_modifier = int(float(self.weight_modifier.get()))
        self.config.time_on_location = int(float(self.time_on_location.get()))
        self.config.take_fly_wings = self.fly_wings_var.get()
        self.config.detect_captcha = self.captcha_var.get()
        self.config.hunt_log_overlay = self.overlay_var.get()
        self.config.skill_button = self.skill_button.get().strip()
        raw = self.skill_delay.get().strip()
        self.config.skill_delay = int(raw) if raw else 0
        self.config.teleport_button = self.teleport_button.get().strip()
        self.config.save_point_button = self.save_point_button.get().strip()
        self.config.open_storage_button = self.open_storage_button.get().strip()
        self.config.skill_timer_button = self.skill_timer_button.get().strip()
        raw_timer = self.skill_timer_interval.get().strip()
        self.config.skill_timer_interval = int(raw_timer) if raw_timer else 0
        self.config.sp_button = self.sp_button.get().strip()
        raw = self.fly_wings_amount.get().strip()
        self.config.fly_wings_amount = int(raw) if raw else 0

    # ══════════════════════════════════════════════════════════════════
    #  BOT LIFECYCLE (thin wrappers that delegate to lifecycle manager)
    # ══════════════════════════════════════════════════════════════════

    def toggle_bot(self) -> None:
        """Called by F12 hotkey or Start/Stop button."""
        if self.lifecycle.state in (
            BotState.RUNNING,
            BotState.PAUSED,
            BotState.STARTING,
        ):
            self.stop_bot()
        else:
            self.start_bot()

    def start_bot(self) -> None:
        """Validate preconditions, sync config, then delegate to lifecycle."""
        try:
            self._start_bot_impl()
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self.log_pipe.log(f"[ERROR] Bot start failed: {exc}")
            messagebox.showerror(
                "Bot Start Error",
                f"Failed to start bot:\n\n{exc}\n\nSee log for traceback.",
            )
            # Also print to console if available
            try:
                print(tb, flush=True)
            except OSError:
                pass

    def _start_bot_impl(self) -> None:
        """Internal bot start logic — wrapped by start_bot() for error handling."""
        if not self.lifecycle.input_ready:
            messagebox.showerror(
                "Error",
                "VIIPER is not ready yet.\nPlease wait for initialization to finish.",
            )
            return
        self.on_window_selected()
        if not self.config.window_id or not window_exists(self.config.window_id):
            messagebox.showerror(
                "Error",
                "Please select a valid game window first.\n"
                "Choose the game in the dropdown and click Refresh if needed.",
            )
            return

        self._sync_config_from_ui()
        self.config.save()

        if not self.lifecycle.start(
            config_snapshot=self.config,
            session_id=self.session.session_id,
        ):
            messagebox.showerror(
                "Error",
                "Bot is already starting or running.",
            )
            return
        self.log_pipe.log("Starting hunt runtime...")

    def stop_bot(self) -> None:
        """Stop the bot (delegates to lifecycle)."""
        self.lifecycle.stop()
        self.log_pipe.log("Bot stopped (VIIPER still running)")

    def pause_bot(self) -> None:
        """Pause the bot (delegates to lifecycle)."""
        self.lifecycle.pause()

    def resume_bot(self) -> None:
        """Resume the bot and restore game window focus."""
        restore_and_activate(self.config.window_id)
        self.lifecycle.resume()
        self.log_pipe.log("Bot resumed")

    # ══════════════════════════════════════════════════════════════════
    #  CALLBACKS (registered with lifecycle and log pipe)
    # ══════════════════════════════════════════════════════════════════

    def _enable_after_viiper(self) -> None:
        """Enable UI widgets after VIIPER is ready (runs on main thread)."""
        self.window_combo.configure(state="readonly")
        self.bot_button.configure(state=tk.NORMAL)
        self.log_pipe.log("All set — select or launch the game window")
        self.refresh_windows()

    def _on_bot_state_changed(self, state: BotState) -> None:
        """Update UI widgets to reflect the current bot state."""
        if state == BotState.RUNNING:
            self.bot_button.configure(text="Stop Bot")
            self.continue_button.configure(state=tk.DISABLED)
            self.bot_status.configure(text="Status: ON")
            self.status_indicator.configure(text="  ON  ", bg="#2e7d32")
            self._lock_ui(True)
        elif state == BotState.STARTING:
            self.bot_button.configure(text="Stop Bot")
            self.continue_button.configure(state=tk.DISABLED)
            self.bot_status.configure(text="Status: Starting...")
            self.status_indicator.configure(text=" START ", bg="#1565c0")
            self._lock_ui(True)
        elif state == BotState.PAUSED:
            self.bot_status.configure(text="Status: PAUSED (TAB)")
            self.status_indicator.configure(text=" PAUSED ", bg="#f9a825")
            self.continue_button.configure(state=tk.NORMAL)
        elif state == BotState.OFF:
            self.bot_button.configure(text="Start Bot")
            self.continue_button.configure(state=tk.DISABLED)
            self.bot_status.configure(text="Status: Off")
            self.status_indicator.configure(text="  OFF  ", bg="#c62828")
            self._lock_ui(False)

    def _maybe_pipe_to_overlay(self, message: str) -> None:
        """Forward log lines to the hunt overlay when the bot is active."""
        if self.lifecycle.state != BotState.OFF:
            self._hunt_overlay.append_log(message, message)
    def _lock_ui(self, locked: bool) -> None:
        """Enable/disable configuration widgets when bot is running."""
        state = tk.DISABLED if locked else tk.NORMAL
        readonly = "disabled" if locked else "readonly"
        self.window_combo.configure(state=readonly)
        self.client_combo.configure(state=readonly)
        self.search_scale.configure(state=state)
        self.weight_scale.configure(state=state)
        for widget in (
            self.skill_button,
            self.skill_delay,
            self.teleport_button,
            self.save_point_button,
            self.open_storage_button,
            self.skill_timer_button,
            self.skill_timer_interval,
            self.sp_button,
            self.fly_wings_amount,
        ):
            widget.configure(state=state)

    # ══════════════════════════════════════════════════════════════════
    #  SHUTDOWN
    # ══════════════════════════════════════════════════════════════════

    def on_exit(self) -> None:
        """Clean shutdown of bot, VIIPER, hotkey, and session."""
        if self.lifecycle.state != BotState.OFF:
            self.stop_bot()
        self.lifecycle.await_shutdown(timeout=DEFAULT_STOP_JOIN_TIMEOUT_S + 1.0)
        self.log_pipe.log("Closing bot and stopping VIIPER...")
        self.viiper.shutdown()
        self.session.end("user exit")
        self.hotkey_manager.destroy()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    # Build/refresh descriptors before the main GUI so hunt never races a rebuild.
    if not preload_mob_descriptors():
        return
    MainWindow().run()


if __name__ == "__main__":
    main()
