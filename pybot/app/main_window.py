"""ViiperHexBots main window (tkinter) — UI building and callback wiring only.

Lifecycle logic             → :mod:`pybot.app.bot_lifecycle`
Hotkey registration/polling → :mod:`pybot.app.hotkey_manager`
Thread-safe log dispatch    → :mod:`pybot.app.log_pipe`
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from pybot.app.bot_lifecycle import BotLifecycleManager, BotState
from pybot.app.bot_controller import DEFAULT_STOP_JOIN_TIMEOUT_S
from pybot.app.config_store import AppConfig, list_client_profiles
from pybot.app.hotkey_manager import HotkeyManager
from pybot.app.log_pipe import LogPipe
from pybot.app.overlay import StatusPanelOverlay, Win32HuntOverlay
from pybot.game_state import GameMemoryPoller, MemorySnapshot
from pybot.app.session_log import AppSessionLog
from pybot.app.startup_splash import preload_mob_descriptors
from pybot.app.viiper_manager import ViiperManager
from pybot.app.win32_util import (
    client_rect_screen,
    enum_game_windows,
    is_window_active,
    restore_and_activate,
    window_exists,
)
from pybot.mobs.import_mob import (
    MobImportError,
    import_mob_from_paths,
    mob_assets_exist,
    resolve_spr_act_paths,
)
from pybot.config.clients import load_client_profile, memory_reading_enabled
from pybot.app.storage_chain_dialog import (
    StorageChainDialog,
    format_storage_chain_summary,
)
from pybot.config.schema import MAX_SKILL_TIMERS, KeyChainStep, SkillTimerSetting
from pybot.mobs.catalog import load_mob_catalog
from pybot.runtime.input.scan_codes import keysym_to_key_name
from pybot.recognition.capture import capture_region
from pybot.recognition.ui.status_panel import (
    StatusPanelValues,
    find_status_panel,
    read_status_panel,
    read_status_panel_currents,
)

MEMORY_POLL_MS = 500
# Searching for the Basic Info header.
STATUS_PANEL_SEARCH_MS = 1000
# Panel locked — read current SP / Weight only.
STATUS_PANEL_VALUE_MS = 200
# Re-parse max SP / Weight this often while the panel stays locked.
STATUS_PANEL_MAX_REFRESH_S = 1.0


class MainWindow:
    """Build the tkinter UI and wire lifecycle/log/hotkey managers together."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Hex Bot")
        # Initial size; replaced by _fit_window_to_content() after widgets exist.
        self.root.geometry("1040x900")
        self.root.minsize(980, 820)

        # ── Data layer ──────────────────────────────────────────────
        self.config = AppConfig().load()
        self.mob_catalog = load_mob_catalog(ensure_assets=False)
        self._check_mob_catalog()
        self.session = AppSessionLog()
        self._hunt_overlay = Win32HuntOverlay()
        self._status_panel_overlay = StatusPanelOverlay()

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
            value=min(max(1, self.config.selected_monster), max(1, len(self.mob_catalog)))
            if self.mob_catalog
            else 1
        )
        self._memory_poller = GameMemoryPoller()
        self._memory_poll_after_id: str | None = None
        self._status_panel_poll_after_id: str | None = None
        self._status_panel_confirmed: StatusPanelValues | None = None
        self._status_panel_max_read_at = 0.0
        # Last known Basic Info origin — anchors the open-panel prompt.
        self._status_panel_anchor: tuple[int, int] = (0, 0)
        # Ignore widget callbacks while building; enable at end of _build_ui.
        self._settings_apply_enabled = False
        self._mob_radios: list[ttk.Radiobutton] = []
        self._settings_checkbuttons: list[ttk.Checkbutton] = []
        self._mob_import_busy = False
        self._mob_radio_frame: ttk.Frame | None = None

        # Build UI (widgets created here, references shared to managers)
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        self._hook_mob_drop_zone()
        self._fit_window_to_content()

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
            # Drop zone can add the first mob; do not exit the app.
            pass

    def _fit_window_to_content(self) -> None:
        """Grow the window to the UI's natural size; block shrink below that."""
        self.root.update_idletasks()
        children = self.root.winfo_children()
        if children:
            content = children[0]
            content.update_idletasks()
            width = content.winfo_reqwidth() + 24
            height = content.winfo_reqheight() + 48
        else:
            width = self.root.winfo_reqwidth()
            height = self.root.winfo_reqheight()
        width = max(width, 980)
        height = max(height, 820)

        min_w, min_h = self.root.minsize()
        width = max(width, min_w)
        height = max(height, min_h)
        self.root.minsize(width, height)

        cur_w = self.root.winfo_width()
        cur_h = self.root.winfo_height()
        if cur_w < 50 or cur_h < 50:
            self.root.geometry(f"{width}x{height}")
            return
        new_w = max(cur_w, width)
        new_h = max(cur_h, height)
        if new_w != cur_w or new_h != cur_h:
            self.root.geometry(f"{new_w}x{new_h}")

    def _rebuild_mob_radio_buttons(self) -> None:
        frame = self._mob_radio_frame
        if frame is None:
            return
        for radio in self._mob_radios:
            radio.destroy()
        self._mob_radios.clear()
        for index, mob in enumerate(self.mob_catalog, start=1):
            radio = ttk.Radiobutton(
                frame,
                text=mob.display_name,
                variable=self.mob_var,
                value=index,
                command=self._apply_ui_settings,
            )
            radio.grid(row=index - 1, column=0, sticky="w")
            self._mob_radios.append(radio)
        if self.mob_catalog:
            current = int(self.mob_var.get() or 1)
            if current < 1 or current > len(self.mob_catalog):
                self.mob_var.set(1)

    def _refresh_mob_radios(self, *, select_stem: str | None = None) -> None:
        self.mob_catalog = load_mob_catalog(ensure_assets=False)
        self.lifecycle._mob_catalog = self.mob_catalog
        self._rebuild_mob_radio_buttons()
        if select_stem and self.mob_catalog:
            key = select_stem.lower()
            for index, mob in enumerate(self.mob_catalog, start=1):
                if mob.descriptor_name.lower() == key:
                    self.mob_var.set(index)
                    break
        if self._settings_apply_enabled:
            self._apply_ui_settings()
        self._fit_window_to_content()

    def _hook_mob_drop_zone(self) -> None:
        try:
            import windnd
        except ImportError:
            self._mob_drop_status.configure(
                text="Install windnd for drag-drop (Browse still works)"
            )
            return

        def _decode_drop_path(item: object) -> Path:
            if isinstance(item, bytes):
                text = item.split(b"\0", 1)[0].decode("utf-8", errors="surrogateescape")
            else:
                text = str(item).split("\0", 1)[0]
            return Path(text)

        def _on_drop(files) -> None:
            paths = [_decode_drop_path(f) for f in files]
            if not paths:
                return
            self.root.after(0, lambda p=paths: self._begin_mob_import(p))

        # Keep a reference so the closure is not collected while hooked.
        self._mob_drop_handler = _on_drop
        self.root.update_idletasks()
        # Child Label HWNDs often never receive WM_DROPFILES; hook the root.
        windnd.hook_dropfiles(self.root, func=_on_drop, force_unicode=True)

    def _browse_mob_assets(self) -> None:
        if not self._can_import_mob():
            return
        paths = filedialog.askopenfilenames(
            title="Select .spr and .act (or cancel and pick a folder)",
            filetypes=[
                ("SPR/ACT", "*.spr *.act"),
                ("SPR", "*.spr"),
                ("ACT", "*.act"),
                ("All", "*.*"),
            ],
        )
        if paths:
            self._begin_mob_import([Path(p) for p in paths])
            return
        folder = filedialog.askdirectory(title="Select folder containing .spr + .act")
        if folder:
            self._begin_mob_import([Path(folder)])

    def _can_import_mob(self) -> bool:
        if self._mob_import_busy:
            messagebox.showinfo("Import mob", "A mob import is already running.")
            return False
        if self.lifecycle.state != BotState.OFF:
            messagebox.showwarning(
                "Import mob",
                "Stop the bot before adding a mob descriptor.",
            )
            return False
        return True

    def _begin_mob_import(self, paths: list[Path]) -> None:
        if not self._can_import_mob():
            return
        try:
            spr, act = resolve_spr_act_paths(paths)
        except MobImportError as exc:
            messagebox.showerror("Import mob", str(exc))
            self._mob_drop_status.configure(text=str(exc))
            return

        stem = spr.stem.lower()
        overwrite = False
        if mob_assets_exist(stem):
            ok = messagebox.askyesno(
                "Import mob",
                f"Mob '{stem}' already exists.\nReplace SPR/ACT and rebuild descriptor?",
            )
            if not ok:
                return
            overwrite = True

        self._mob_import_busy = True
        self._mob_browse_button.configure(state=tk.DISABLED)
        self._mob_drop_status.configure(text=f"Building {stem}…")
        self.log_pipe.log(f"[MOB] importing {spr.name} + {act.name}")

        def _worker() -> None:
            try:
                entry = import_mob_from_paths([spr, act], overwrite=overwrite)
            except Exception as exc:
                err = exc
                self.root.after(0, lambda: self._mob_import_failed(err))
                return
            stem_ready = entry.descriptor_name
            self.root.after(0, lambda: self._mob_import_succeeded(stem_ready))

        threading.Thread(target=_worker, name="mob-import", daemon=True).start()

    def _mob_import_failed(self, exc: Exception) -> None:
        self._mob_import_busy = False
        if self.lifecycle.state == BotState.OFF:
            self._mob_browse_button.configure(state=tk.NORMAL)
        self._mob_drop_status.configure(text=f"Failed: {exc}")
        self.log_pipe.log(f"[MOB] import failed: {exc}")
        messagebox.showerror("Import mob", f"Failed to build descriptor:\n\n{exc}")

    def _mob_import_succeeded(self, stem: str) -> None:
        self._mob_import_busy = False
        if self.lifecycle.state == BotState.OFF:
            self._mob_browse_button.configure(state=tk.NORMAL)
        self._refresh_mob_radios(select_stem=stem)
        self._mob_drop_status.configure(text=f"Ready: {stem}")
        self.log_pipe.log(f"[MOB] descriptor ready: {stem}")
        messagebox.showinfo("Import mob", f"Descriptor built for '{stem}'.")

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

        # ── Status & Input (two-column side panel) ──────────────────
        status_input_frame = ttk.LabelFrame(main, text="Status & Input", padding=8)
        status_input_frame.grid(row=1, column=2, sticky="nsew")
        status_input_frame.columnconfigure(0, weight=0)
        status_input_frame.columnconfigure(2, weight=1)

        status_col = ttk.Frame(status_input_frame)
        status_col.grid(row=0, column=0, sticky="nw", padx=(0, 10))
        ttk.Label(
            status_col, text="Status", font=("Segoe UI", 9, "bold")
        ).pack(anchor="w")
        self.bot_status = ttk.Label(status_col, text="Off")
        self.bot_status.pack(anchor="w", pady=(2, 0))
        self.status_indicator = tk.Label(
            status_col,
            text="  OFF  ",
            bg="#c62828",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            width=10,
        )
        self.status_indicator.pack(anchor="w", pady=(6, 8))
        self.input_status = ttk.Label(status_col, text="Input: Starting...")
        self.input_status.pack(anchor="w")
        self.input_hint = ttk.Label(
            status_col,
            text="Launch the game after VIIPER is ready",
            wraplength=140,
        )
        self.input_hint.pack(anchor="w", pady=(2, 0))

        ttk.Separator(status_input_frame, orient=tk.VERTICAL).grid(
            row=0, column=1, sticky="ns", padx=4
        )

        profile_col = ttk.Frame(status_input_frame)
        profile_col.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
        profile_col.columnconfigure(1, weight=1)
        ttk.Label(
            profile_col, text="Client Profile", font=("Segoe UI", 9, "bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        self.client_combo = ttk.Combobox(
            profile_col,
            values=list_client_profiles(),
            state="readonly",
            width=16,
        )
        self.client_combo.set(self.config.client_profile)
        self.client_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 4))
        self.client_combo.bind("<<ComboboxSelected>>", self.on_client_changed)

        ttk.Separator(profile_col, orient=tk.HORIZONTAL).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(6, 6)
        )
        ttk.Label(profile_col, text="Name:").grid(row=3, column=0, sticky="w")
        self.memory_name = ttk.Label(profile_col, text="—")
        self.memory_name.grid(row=3, column=1, sticky="w", padx=(8, 0))
        ttk.Label(profile_col, text="HP:").grid(row=4, column=0, sticky="w", pady=(2, 0))
        self.memory_hp = ttk.Label(profile_col, text="—")
        self.memory_hp.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(2, 0))
        ttk.Label(profile_col, text="SP:").grid(row=5, column=0, sticky="w", pady=(2, 0))
        self.memory_sp = ttk.Label(profile_col, text="—")
        self.memory_sp.grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(2, 0))
        ttk.Label(profile_col, text="Weight:").grid(
            row=6, column=0, sticky="w", pady=(2, 0)
        )
        self.memory_weight = ttk.Label(profile_col, text="—")
        self.memory_weight.grid(row=6, column=1, sticky="w", padx=(8, 0), pady=(2, 0))

        # ── Setup ──────────────────────────────────────────────────
        setup_frame = ttk.LabelFrame(main, text="Setup", padding=8)
        setup_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 8), pady=(8, 0))
        setup_frame.columnconfigure(1, weight=1)

        mob_col = ttk.Frame(setup_frame)
        mob_col.grid(row=0, column=0, sticky="nw")
        ttk.Label(mob_col, text="Descriptor Mob:").grid(row=0, column=0, sticky="w")
        self._mob_radio_frame = ttk.Frame(mob_col)
        self._mob_radio_frame.grid(row=1, column=0, sticky="nw")
        self._rebuild_mob_radio_buttons()

        drop_frame = ttk.LabelFrame(mob_col, text="Add mob (SPR + ACT)", padding=6)
        drop_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self._mob_drop_label = tk.Label(
            drop_frame,
            text="Drop .spr + .act here\n(or a folder with both)",
            relief=tk.GROOVE,
            borderwidth=2,
            width=28,
            height=3,
            justify=tk.CENTER,
            background="#f0f0f0",
        )
        self._mob_drop_label.grid(row=0, column=0, sticky="ew")
        drop_btns = ttk.Frame(drop_frame)
        drop_btns.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._mob_browse_button = ttk.Button(
            drop_btns,
            text="Browse…",
            command=self._browse_mob_assets,
        )
        self._mob_browse_button.pack(side=tk.LEFT)
        self._mob_drop_status = ttk.Label(drop_frame, text="", wraplength=200)
        self._mob_drop_status.grid(row=2, column=0, sticky="w", pady=(4, 0))

        mode_col = ttk.Frame(setup_frame)
        mode_col.grid(row=0, column=1, sticky="nw", padx=(16, 0))

        mode_row = ttk.Frame(mode_col)
        mode_row.grid(row=0, column=0, sticky="w")
        ttk.Label(mode_row, text="Hunt Mode:").pack(side=tk.LEFT)
        self.hunt_mode_var = tk.StringVar(value=self.config.hunt_mode)
        self.hunt_mode_combo = ttk.Combobox(
            mode_row,
            textvariable=self.hunt_mode_var,
            values=("teleport", "hybrid", "walk"),
            state="readonly",
            width=12,
        )
        self.hunt_mode_combo.pack(side=tk.LEFT, padx=(6, 0))
        self.hunt_mode_combo.bind("<<ComboboxSelected>>", self._apply_ui_settings)

        ttk.Label(mode_col, text="Search Range (9-16 cells):").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        search_row = ttk.Frame(mode_col)
        search_row.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        self.search_range = tk.IntVar(value=self.config.search_range)
        self.search_scale = ttk.Scale(
            search_row,
            from_=9,
            to=16,
            orient=tk.HORIZONTAL,
            variable=self.search_range,
            command=self._update_search_label,
        )
        self.search_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.search_label = ttk.Label(search_row, text=str(self.config.search_range))
        self.search_label.pack(side=tk.LEFT, padx=(6, 0))

        # ── Keybindings (spans remaining middle-row width) ───────────
        keys_frame = ttk.LabelFrame(main, text="Keybindings", padding=8)
        keys_frame.grid(
            row=2, column=1, columnspan=2, sticky="nsew", pady=(8, 0)
        )
        keys_frame.columnconfigure(0, weight=1)
        keys_frame.columnconfigure(2, weight=0)

        keys_main = ttk.Frame(keys_frame)
        keys_main.grid(row=0, column=0, sticky="nw")

        self.skill_button = self._key_entry(
            keys_main,
            "Attack Skill Key:",
            self.config.skill_button,
            0,
            0,
            capture_key=True,
        )
        self.skill_delay = self._key_entry(
            keys_main,
            "Attack Delay:",
            str(self.config.skill_delay or 500),
            0,
            1,
            width=7,
        )
        tp_row = ttk.Frame(keys_main)
        tp_row.grid(row=1, column=0, sticky="w", pady=2)
        ttk.Label(tp_row, text="Teleport Key:").pack(side=tk.LEFT)
        self.teleport_button = ttk.Entry(tp_row, width=6)
        self.teleport_button.insert(0, self.config.teleport_button)
        self.teleport_button.pack(side=tk.LEFT, padx=(4, 0))
        self._bind_key_capture(self.teleport_button)
        self._bind_setting_entry(self.teleport_button)
        ttk.Label(tp_row, text="Creamy TP Key:").pack(side=tk.LEFT, padx=(12, 0))
        self.creamy_tp_button = ttk.Entry(tp_row, width=6)
        self.creamy_tp_button.insert(0, self.config.creamy_tp_button)
        self.creamy_tp_button.pack(side=tk.LEFT, padx=(4, 0))
        self._bind_key_capture(self.creamy_tp_button)
        self._bind_setting_entry(self.creamy_tp_button)
        self.teleport_delay = self._key_entry(
            keys_main,
            "Teleport Delay:",
            str(self.config.teleport_delay or 800),
            1,
            1,
            width=7,
        )
        self.save_point_button = self._key_entry(
            keys_main,
            "To SavePoint Key:",
            self.config.save_point_button,
            2,
            0,
            capture_key=True,
        )
        hp_cell = ttk.Frame(keys_main)
        hp_cell.grid(row=3, column=0, sticky="w", pady=2)
        ttk.Label(hp_cell, text="HP Restore Key:").pack(side=tk.LEFT)
        self.hp_button = ttk.Entry(hp_cell, width=6)
        self.hp_button.insert(0, self.config.hp_button)
        self.hp_button.pack(side=tk.LEFT, padx=(4, 0))
        self._bind_key_capture(self.hp_button)
        self._bind_setting_entry(self.hp_button)
        self.heal_skill_var = tk.BooleanVar(value=self.config.heal_skill)
        heal_check = ttk.Checkbutton(
            hp_cell,
            text="Heal skill",
            variable=self.heal_skill_var,
            command=self._apply_ui_settings,
        )
        heal_check.pack(side=tk.LEFT, padx=(8, 0))
        self._settings_checkbuttons.append(heal_check)
        self.sp_button = self._key_entry(
            keys_main,
            "SP Item Key:",
            self.config.sp_button,
            4,
            0,
            capture_key=True,
        )
        sit_cell = ttk.Frame(keys_main)
        sit_cell.grid(row=5, column=0, sticky="w", pady=2)
        ttk.Label(sit_cell, text="Sit On Low Sp Key:").pack(side=tk.LEFT)
        self.sit_on_low_sp_button = ttk.Entry(sit_cell, width=6)
        self.sit_on_low_sp_button.insert(
            0, self.config.sit_on_low_sp_button or "insert"
        )
        self.sit_on_low_sp_button.pack(side=tk.LEFT, padx=(4, 0))
        self._bind_key_capture(self.sit_on_low_sp_button)
        self.sit_on_low_sp_var = tk.BooleanVar(value=self.config.sit_on_low_sp)
        self.sit_on_low_sp_toggle = tk.Button(
            sit_cell,
            text="On" if self.config.sit_on_low_sp else "Off",
            width=4,
            relief=tk.RAISED,
            command=self._toggle_sit_on_low_sp,
        )
        self.sit_on_low_sp_toggle.pack(side=tk.LEFT, padx=(4, 0))
        self._refresh_sit_toggle()
        storage_cell = ttk.Frame(keys_main)
        storage_cell.grid(row=6, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(storage_cell, text="Open Storage:").pack(side=tk.LEFT)
        self.open_storage_cog = ttk.Button(
            storage_cell,
            text="⚙",
            width=3,
            command=self._open_storage_chain_dialog,
        )
        self.open_storage_cog.pack(side=tk.LEFT, padx=(4, 0))
        self.open_storage_summary = ttk.Label(
            storage_cell,
            text=format_storage_chain_summary(self.config.open_storage_chain),
        )
        self.open_storage_summary.pack(side=tk.LEFT, padx=(6, 0))
        fly_cell = ttk.Frame(keys_main)
        fly_cell.grid(row=7, column=0, sticky="w", pady=2)
        self.fly_wings_var = tk.BooleanVar(value=self.config.take_fly_wings)
        fly_check = ttk.Checkbutton(
            fly_cell,
            text="Take Fly Wings",
            variable=self.fly_wings_var,
            command=self._apply_ui_settings,
        )
        fly_check.pack(side=tk.LEFT)
        self._settings_checkbuttons.append(fly_check)
        self.fly_wings_amount = ttk.Entry(fly_cell, width=6)
        self.fly_wings_amount.insert(0, str(self.config.fly_wings_amount))
        self.fly_wings_amount.pack(side=tk.LEFT, padx=(4, 0))
        self._bind_setting_entry(self.fly_wings_amount)
        weight_cell = ttk.Frame(keys_main)
        weight_cell.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(weight_cell, text="Items to storage weight:").pack(side=tk.LEFT)
        # 49 = Off (AHK); 50–90 = active threshold %.
        initial_weight = max(49, min(90, int(self.config.weight_modifier)))
        self.storage_weight = tk.IntVar(value=initial_weight)
        self.storage_weight_scale = ttk.Scale(
            weight_cell,
            from_=49,
            to=90,
            orient=tk.HORIZONTAL,
            variable=self.storage_weight,
            command=self._update_storage_weight_label,
            length=140,
        )
        self.storage_weight_scale.pack(side=tk.LEFT, padx=(6, 0))
        self.storage_weight_label = ttk.Label(weight_cell, text="")
        self.storage_weight_label.pack(side=tk.LEFT, padx=(6, 0))
        self._update_storage_weight_label()

        ttk.Separator(keys_frame, orient=tk.VERTICAL).grid(
            row=0, column=1, sticky="ns", padx=10
        )

        timer_col = ttk.Frame(keys_frame)
        timer_col.grid(row=0, column=2, sticky="n")
        timer_header = ttk.Frame(timer_col)
        timer_header.grid(row=0, column=0, sticky="w", pady=(0, 4))
        ttk.Label(
            timer_header, text="Timers", font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT)
        self._add_timer_button = ttk.Button(
            timer_header, text="+", width=3, command=self._on_add_timer_box
        )
        self._add_timer_button.pack(side=tk.LEFT, padx=(6, 0))
        self._timer_grid = ttk.Frame(timer_col)
        self._timer_grid.grid(row=1, column=0, sticky="nw")
        self._timer_boxes: list[dict[str, object]] = []
        initial_timers = list(self.config.skill_timers)
        if not initial_timers:
            initial_timers = [SkillTimerSetting()]
        for timer in initial_timers[:MAX_SKILL_TIMERS]:
            self._add_timer_box(timer)
        self._refresh_timer_add_button()

        # ── Log (full width, expands with window) ───────────────────
        log_frame = ttk.LabelFrame(main, text="Log", padding=8)
        log_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
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
        overlay_check = ttk.Checkbutton(
            log_frame,
            text="Hunt log overlay on game",
            variable=self.overlay_var,
            command=self._apply_ui_settings,
        )
        overlay_check.pack(anchor="w", pady=(6, 0))
        self._settings_checkbuttons.append(overlay_check)

        # ── Controls (pinned below log, never clipped) ──────────────
        controls = ttk.Frame(main)
        controls.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(12, 0))
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
        main.rowconfigure(3, weight=1)
        self._sync_memory_reading_from_profile()
        self._update_search_label()
        self._schedule_memory_poll()
        self._schedule_status_panel_poll()
        self._settings_apply_enabled = True

    def _labeled_entry(
        self, parent, label: str, value: str, row: int
    ) -> ttk.Entry:
        cell = ttk.Frame(parent)
        cell.grid(row=row, column=0, sticky="w", pady=2)
        ttk.Label(cell, text=label).pack(side=tk.LEFT)
        entry = ttk.Entry(cell, width=10)
        entry.insert(0, value)
        entry.pack(side=tk.LEFT, padx=(4, 0))
        self._bind_setting_entry(entry)
        return entry

    def _key_entry(
        self,
        parent,
        label: str,
        value: str,
        row: int,
        column: int,
        *,
        width: int = 5,
        capture_key: bool = False,
    ) -> ttk.Entry:
        cell = ttk.Frame(parent)
        cell.grid(
            row=row,
            column=column,
            sticky="w",
            pady=2,
            padx=(0 if column == 0 else 12, 0),
        )
        ttk.Label(cell, text=label).pack(side=tk.LEFT)
        entry = ttk.Entry(cell, width=width)
        entry.insert(0, value)
        entry.pack(side=tk.LEFT, padx=(4, 0))
        if capture_key:
            self._bind_key_capture(entry)
        else:
            self._bind_setting_entry(entry)
        return entry

    def _bind_setting_entry(self, entry: ttk.Entry) -> None:
        """Persist settings when the user finishes editing a text field."""
        entry.bind("<FocusOut>", self._apply_ui_settings)
        entry.bind("<Return>", self._apply_ui_settings)

    def _bind_key_capture(self, entry: ttk.Entry) -> None:
        """Capture the next key press into the entry (supports F-keys)."""
        entry.bind("<KeyPress>", self._on_key_capture)
        entry.bind("<FocusOut>", self._apply_ui_settings)

    def _on_key_capture(self, event: tk.Event) -> str:
        widget = event.widget
        if event.keysym in ("BackSpace", "Delete"):
            widget.delete(0, tk.END)
            self._apply_ui_settings()
            return "break"
        name = keysym_to_key_name(event.keysym)
        if not name:
            return "break"
        widget.delete(0, tk.END)
        widget.insert(0, name)
        self._apply_ui_settings()
        return "break"

    def _refresh_storage_chain_summary(self) -> None:
        self.open_storage_summary.configure(
            text=format_storage_chain_summary(self.config.open_storage_chain)
        )

    def _open_storage_chain_dialog(self) -> None:
        def _apply(steps: list[KeyChainStep]) -> None:
            self.config.open_storage_chain = list(steps)
            self._refresh_storage_chain_summary()
            self._apply_ui_settings()

        StorageChainDialog(
            self.root,
            list(self.config.open_storage_chain),
            on_apply=_apply,
        )

    def _toggle_sit_on_low_sp(self) -> None:
        self.sit_on_low_sp_var.set(not self.sit_on_low_sp_var.get())
        self._refresh_sit_toggle()
        self._apply_ui_settings()

    def _refresh_sit_toggle(self) -> None:
        if self.sit_on_low_sp_var.get():
            self.sit_on_low_sp_toggle.configure(
                text="On",
                bg="#2e7d32",
                fg="white",
                activebackground="#1b5e20",
                activeforeground="white",
            )
        else:
            self.sit_on_low_sp_toggle.configure(
                text="Off",
                bg="#c62828",
                fg="white",
                activebackground="#8e0000",
                activeforeground="white",
            )

    def _add_timer_box(self, timer: SkillTimerSetting | None = None) -> None:
        if len(self._timer_boxes) >= MAX_SKILL_TIMERS:
            return
        timer = timer or SkillTimerSetting()
        index = len(self._timer_boxes)
        row, col = divmod(index, 2)
        box = ttk.LabelFrame(self._timer_grid, text=f"T{index + 1}", padding=3)
        box.grid(row=row, column=col, sticky="nw", padx=3, pady=3)

        ttk.Label(box, text="Key").grid(row=0, column=0, sticky="w")
        key_entry = ttk.Entry(box, width=4)
        key_entry.insert(0, timer.button)
        key_entry.grid(row=0, column=1, sticky="w", padx=(2, 0))
        self._bind_key_capture(key_entry)

        ttk.Label(box, text="s").grid(row=1, column=0, sticky="w", pady=(2, 0))
        delay_entry = ttk.Entry(box, width=4)
        delay_entry.insert(0, str(timer.interval_s))
        delay_entry.grid(row=1, column=1, sticky="w", padx=(2, 0), pady=(2, 0))
        self._bind_setting_entry(delay_entry)

        remove_btn = ttk.Button(
            box,
            text="×",
            width=2,
            command=lambda i=index: self._on_remove_timer_box(i),
        )
        remove_btn.grid(row=0, column=2, rowspan=2, sticky="ne", padx=(4, 0))

        self._timer_boxes.append(
            {
                "frame": box,
                "key": key_entry,
                "delay": delay_entry,
                "remove": remove_btn,
            }
        )
        self._relayout_timer_boxes()
        self._refresh_timer_add_button()

    def _on_add_timer_box(self) -> None:
        self._add_timer_box(SkillTimerSetting())
        self._apply_ui_settings()

    def _on_remove_timer_box(self, index: int) -> None:
        if index < 0 or index >= len(self._timer_boxes):
            return
        # Keep at least one empty slot visible.
        if len(self._timer_boxes) <= 1:
            key = self._timer_boxes[0]["key"]
            delay = self._timer_boxes[0]["delay"]
            assert isinstance(key, ttk.Entry)
            assert isinstance(delay, ttk.Entry)
            key.delete(0, tk.END)
            delay.delete(0, tk.END)
            delay.insert(0, "20")
            self._apply_ui_settings()
            return
        box = self._timer_boxes.pop(index)
        frame = box["frame"]
        assert isinstance(frame, ttk.LabelFrame)
        frame.destroy()
        self._relayout_timer_boxes()
        self._refresh_timer_add_button()
        self._apply_ui_settings()

    def _relayout_timer_boxes(self) -> None:
        for index, box in enumerate(self._timer_boxes):
            frame = box["frame"]
            remove_btn = box["remove"]
            assert isinstance(frame, ttk.LabelFrame)
            assert isinstance(remove_btn, ttk.Button)
            row, col = divmod(index, 2)
            frame.grid(row=row, column=col, sticky="nw", padx=3, pady=3)
            frame.configure(text=f"T{index + 1}")
            remove_btn.configure(command=lambda i=index: self._on_remove_timer_box(i))

    def _refresh_timer_add_button(self) -> None:
        if len(self._timer_boxes) >= MAX_SKILL_TIMERS:
            self._add_timer_button.configure(state=tk.DISABLED)
        else:
            self._add_timer_button.configure(state=tk.NORMAL)

    def _collect_skill_timers_from_ui(self) -> list[SkillTimerSetting]:
        timers: list[SkillTimerSetting] = []
        for box in self._timer_boxes:
            key_entry = box["key"]
            delay_entry = box["delay"]
            assert isinstance(key_entry, ttk.Entry)
            assert isinstance(delay_entry, ttk.Entry)
            button = key_entry.get().strip()
            raw_delay = delay_entry.get().strip()
            interval = int(raw_delay) if raw_delay else 20
            if button:
                timers.append(
                    SkillTimerSetting(button=button, interval_s=max(1, interval))
                )
        return timers[:MAX_SKILL_TIMERS]

    # ══════════════════════════════════════════════════════════════════
    #  UI CALLBACKS (widget value helpers)
    # ══════════════════════════════════════════════════════════════════

    def _update_search_label(self, *_args) -> None:
        cells = int(float(self.search_range.get()))
        px = cells * 64
        self.search_label.configure(text=f"{cells} ({px}px)")
        self.lifecycle.set_search_range_cells(cells)
        self._apply_ui_settings()

    def _update_storage_weight_label(self, *_args) -> None:
        percent = int(float(self.storage_weight.get()))
        if percent < 50:
            self.storage_weight_label.configure(text="Off")
        else:
            self.storage_weight_label.configure(text=f"{percent}%")
        self._apply_ui_settings()

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
        # Index-based: two clients can share title/process; label lookup
        # would keep binding memory to the first duplicate.
        index = self.window_combo.current()
        if index < 0 or index >= len(self.window_entries):
            return
        entry = self.window_entries[index]
        self.config.window_id = entry.hwnd
        self.config.window_title = entry.title
        self.config.window_process = entry.process
        self.config.last_session_title = entry.title
        self.config.last_session_process = entry.process
        self.window_info.configure(text=entry.display_text)
        self._memory_poller.reset()
        self._refresh_memory_stats()
        self._refresh_status_panel_overlay()
        if self._settings_apply_enabled:
            self.config.save()

    def on_client_changed(self, *_event) -> None:
        self.config.client_profile = self.client_combo.get()
        self._sync_memory_reading_from_profile()
        self._memory_poller.reset()
        memory = "on" if self.config.use_memory_reading else "off"
        if self.config.use_memory_reading:
            source = "memory (HP from status panel)"
        else:
            source = "status panel"
        self.log_pipe.log(
            f"Client profile: {self.config.client_profile} "
            f"(memory reading {memory}, stats from {source})"
        )
        self._refresh_memory_stats()
        self._refresh_status_panel_overlay()
        if self._settings_apply_enabled:
            self.config.save()

    def _sync_memory_reading_from_profile(self) -> None:
        """Memory reading follows the profile: Generic off, server profiles on."""
        self.config.use_memory_reading = memory_reading_enabled(self.client_combo.get())

    def _panel_owns_sp_weight(self) -> bool:
        """True when SP/Weight come from Basic Info OCR (Generic / no memory)."""
        return not self.config.use_memory_reading

    @staticmethod
    def _format_pair(current: int | None, maximum: int | None) -> str:
        if current is None and maximum is None:
            return "—"
        if maximum is None:
            return str(current)
        if current is None:
            return f"—/{maximum}"
        return f"{current}/{maximum}"

    def _clear_memory_stats(self, placeholder: str = "—") -> None:
        self.memory_name.configure(text=placeholder)
        self.memory_sp.configure(text=placeholder)
        self.memory_weight.configure(text=placeholder)

    def _clear_vision_stats(self, placeholder: str = "—") -> None:
        """Clear vision-backed labels (HP always; SP/Weight when panel owns them)."""
        self.memory_hp.configure(text=placeholder)
        if self._panel_owns_sp_weight():
            self.memory_sp.configure(text=placeholder)
            self.memory_weight.configure(text=placeholder)

    def _apply_memory_snapshot(self, snap: MemorySnapshot) -> None:
        if not snap.ok:
            self.memory_name.configure(text="—")
            self.memory_sp.configure(text="—")
            self.memory_weight.configure(text="—")
            return
        self.memory_name.configure(text=snap.char_name or "—")
        self.memory_sp.configure(text=self._format_pair(snap.sp, snap.sp_max))
        self.memory_weight.configure(
            text=self._format_pair(snap.weight, snap.weight_max)
        )
        # HP is vision-only — never overwrite from memory polls.

    def _apply_status_panel_stats(self, values: StatusPanelValues) -> None:
        """Apply vision SP/Weight when memory is off (HP set in commit)."""
        if not self._panel_owns_sp_weight():
            return
        self.memory_sp.configure(text=self._format_pair(values.sp, values.sp_max))
        self.memory_weight.configure(
            text=self._format_pair(values.weight, values.weight_max)
        )

    def _refresh_memory_stats(self) -> None:
        if not self.config.use_memory_reading:
            # Generic: Name has no memory source; SP/Weight come from vision.
            self.memory_name.configure(text="—")
            return
        profile = load_client_profile(self.config.client_profile)
        if profile is None or not profile.memory.has_any:
            self.memory_name.configure(text="—")
            self.memory_sp.configure(text="—")
            self.memory_weight.configure(text="—")
            return
        hwnd = self.config.window_id
        if not hwnd or not window_exists(hwnd):
            self.memory_name.configure(text="—")
            self.memory_sp.configure(text="—")
            self.memory_weight.configure(text="—")
            return
        snap = self._memory_poller.read(hwnd, profile.memory)
        self._apply_memory_snapshot(snap)

    def _schedule_memory_poll(self) -> None:
        if self._memory_poll_after_id is not None:
            try:
                self.root.after_cancel(self._memory_poll_after_id)
            except tk.TclError:
                pass
            self._memory_poll_after_id = None

        def _tick() -> None:
            self._memory_poll_after_id = None
            try:
                self._refresh_memory_stats()
            finally:
                if self.root.winfo_exists():
                    self._memory_poll_after_id = self.root.after(
                        MEMORY_POLL_MS, _tick
                    )

        self._memory_poll_after_id = self.root.after(MEMORY_POLL_MS, _tick)

    def _clear_status_panel_ui(self) -> None:
        """Panel missing/unreadable — drop HP; drop SP/Weight only if vision owns them."""
        self._clear_vision_stats()

    def _reset_status_panel_tracking(self) -> None:
        self._status_panel_confirmed = None
        self._status_panel_max_read_at = 0.0

    @staticmethod
    def _status_panel_numbers(
        values: StatusPanelValues,
    ) -> tuple[int, int, int, int, int | None, int | None]:
        return (
            values.hp,
            values.hp_max,
            values.sp,
            values.sp_max,
            values.weight,
            values.weight_max,
        )

    def _show_panel_missing(
        self,
        *,
        client_left: int,
        client_top: int,
    ) -> None:
        """Basic Info not open — clear reads and prompt to open it."""
        self._reset_status_panel_tracking()
        self._clear_status_panel_ui()
        self._status_panel_overlay.show_panel_missing(
            client_left=client_left,
            client_top=client_top,
            panel_origin=self._status_panel_anchor,
        )

    def _commit_status_panel(
        self,
        values: StatusPanelValues,
        *,
        client_left: int,
        client_top: int,
    ) -> None:
        """Store a successful read; UI stats update only when numbers change."""
        previous = self._status_panel_confirmed
        self._status_panel_confirmed = values
        self._status_panel_anchor = values.panel_origin
        self._status_panel_overlay.update(
            values, client_left=client_left, client_top=client_top
        )
        # HP is vision-only — always mirror into the bot UI from panel OCR.
        self.memory_hp.configure(text=self._format_pair(values.hp, values.hp_max))
        if previous is not None and self._status_panel_numbers(
            previous
        ) == self._status_panel_numbers(values):
            return
        if self._panel_owns_sp_weight():
            self._apply_status_panel_stats(values)

    def _refresh_status_panel_overlay(self) -> int:
        """Each tick: find Basic Info header, then read or show open-panel.

        HP is always OCR'd on the full-refresh cadence (including when memory
        supplies SP/Weight). SP/Weight currents still refresh every tick.

        - Header missing → clear vision-backed labels, show open-panel prompt
        - Header found → read values and show under the panel
        - Full current+max (incl. HP OCR) every ``STATUS_PANEL_MAX_REFRESH_S``

        Overlay stays visible (no hide/show flash). Both states use the same
        slot under the panel so header/digit ROIs stay uncovered.
        """
        hwnd = self.config.window_id
        if not hwnd or not window_exists(hwnd) or not is_window_active(hwnd):
            self._status_panel_overlay.hide()
            return STATUS_PANEL_SEARCH_MS

        client = client_rect_screen(hwnd)
        if client is None:
            self._reset_status_panel_tracking()
            self._status_panel_overlay.hide()
            self._clear_status_panel_ui()
            return STATUS_PANEL_SEARCH_MS

        left, top, width, height = client

        frame = capture_region(left, top, width, height)
        if frame is None or frame.size == 0:
            self._show_panel_missing(client_left=left, client_top=top)
            return STATUS_PANEL_SEARCH_MS

        origin = find_status_panel(frame)
        if origin is None:
            self._show_panel_missing(client_left=left, client_top=top)
            return STATUS_PANEL_SEARCH_MS

        confirmed = self._status_panel_confirmed
        now = time.monotonic()
        refresh_max = (
            confirmed is None
            or now - self._status_panel_max_read_at >= STATUS_PANEL_MAX_REFRESH_S
        )
        if refresh_max:
            values = read_status_panel(frame, origin=origin)
        else:
            values = read_status_panel_currents(
                frame,
                origin,
                hp=confirmed.hp,
                hp_max=confirmed.hp_max,
                sp_max=confirmed.sp_max,
                weight_max=confirmed.weight_max,
            )
        if values is None:
            self._show_panel_missing(client_left=left, client_top=top)
            return STATUS_PANEL_SEARCH_MS

        if refresh_max:
            self._status_panel_max_read_at = now
        self._commit_status_panel(values, client_left=left, client_top=top)
        return STATUS_PANEL_VALUE_MS

    def _schedule_status_panel_poll(self) -> None:
        if self._status_panel_poll_after_id is not None:
            try:
                self.root.after_cancel(self._status_panel_poll_after_id)
            except tk.TclError:
                pass
            self._status_panel_poll_after_id = None

        def _tick() -> None:
            self._status_panel_poll_after_id = None
            delay = STATUS_PANEL_SEARCH_MS
            try:
                delay = self._refresh_status_panel_overlay()
            finally:
                if self.root.winfo_exists():
                    self._status_panel_poll_after_id = self.root.after(
                        max(50, int(delay)), _tick
                    )

        self._status_panel_poll_after_id = self.root.after(
            STATUS_PANEL_SEARCH_MS, _tick
        )

    def _sync_config_from_ui(self) -> None:
        """Read all UI widget values into self.config."""
        self.config.client_profile = self.client_combo.get()
        self._sync_memory_reading_from_profile()
        self.config.selected_monster = self.mob_var.get()
        self.config.hunt_mode = self.hunt_mode_var.get()
        self.config.search_range = int(float(self.search_range.get()))
        self.config.take_fly_wings = self.fly_wings_var.get()
        self.config.hunt_log_overlay = self.overlay_var.get()
        self.config.skill_button = self.skill_button.get().strip()
        raw = self.skill_delay.get().strip()
        self.config.skill_delay = int(raw) if raw else 500
        self.config.teleport_button = self.teleport_button.get().strip()
        self.config.creamy_tp_button = self.creamy_tp_button.get().strip()
        raw_tp = self.teleport_delay.get().strip()
        self.config.teleport_delay = int(raw_tp) if raw_tp else 800
        self.config.save_point_button = self.save_point_button.get().strip()
        # open_storage_chain is edited via the cog dialog
        self.config.weight_modifier = int(float(self.storage_weight.get()))
        self.config.skill_timers = self._collect_skill_timers_from_ui()
        self.config.hp_button = self.hp_button.get().strip()
        self.config.heal_skill = self.heal_skill_var.get()
        self.config.sp_button = self.sp_button.get().strip()
        self.config.sit_on_low_sp_button = self.sit_on_low_sp_button.get().strip()
        self.config.sit_on_low_sp = self.sit_on_low_sp_var.get()
        raw = self.fly_wings_amount.get().strip()
        self.config.fly_wings_amount = int(raw) if raw else 0

    def _apply_ui_settings(self, *_args) -> None:
        """Push current GUI values into config.ini as soon as they change."""
        if not self._settings_apply_enabled:
            return
        try:
            self._sync_config_from_ui()
            self.config.save()
        except ValueError:
            # Incomplete numeric field while typing; wait for a valid edit.
            return

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
                parent=self.root,
            )
            return
        self.on_window_selected()
        if not self.config.window_id or not window_exists(self.config.window_id):
            messagebox.showerror(
                "Error",
                "Please select a valid game window first.\n"
                "Choose the game in the dropdown and click Refresh if needed.",
                parent=self.root,
            )
            return

        try:
            self._sync_config_from_ui()
            self.config.save()
        except ValueError as exc:
            messagebox.showerror(
                "Invalid Settings",
                f"Fix numeric fields before starting:\n\n{exc}",
                parent=self.root,
            )
            return

        from datetime import datetime

        # Fresh session id each start so restart logs are not mixed with the
        # previous hunt and file handlers stay unambiguous on Windows.
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if not self.lifecycle.start(
            config_snapshot=self.config,
            session_id=session_id,
        ):
            messagebox.showerror(
                "Error",
                "Bot is already starting or running.\n"
                "If it is stuck on Starting, press Stop once, then Start again.",
                parent=self.root,
            )
            return
        self.log_pipe.log(f"Starting hunt runtime... session={session_id}")

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
            self.bot_status.configure(text="On")
            self.status_indicator.configure(text="  ON  ", bg="#2e7d32")
            self._lock_ui(True)
        elif state == BotState.STARTING:
            self.bot_button.configure(text="Stop Bot")
            self.continue_button.configure(state=tk.DISABLED)
            self.bot_status.configure(text="Starting...")
            self.status_indicator.configure(text=" START ", bg="#1565c0")
            self._lock_ui(True)
        elif state == BotState.PAUSED:
            self.bot_button.configure(text="Stop Bot")
            self.bot_status.configure(text="Paused (TAB)")
            self.status_indicator.configure(text=" PAUSED ", bg="#f9a825")
            self.continue_button.configure(state=tk.NORMAL)
        elif state == BotState.OFF:
            self.bot_button.configure(text="Start Bot")
            self.continue_button.configure(state=tk.DISABLED)
            self.bot_status.configure(text="Off")
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
        self.hunt_mode_combo.configure(state=readonly)
        self.search_scale.configure(state=state)
        self.storage_weight_scale.configure(state=state)
        for radio in self._mob_radios:
            radio.configure(state=state)
        if hasattr(self, "_mob_browse_button"):
            browse_state = tk.DISABLED if (locked or self._mob_import_busy) else tk.NORMAL
            self._mob_browse_button.configure(state=browse_state)
        for check in self._settings_checkbuttons:
            check.configure(state=state)
        for widget in (
            self.skill_button,
            self.skill_delay,
            self.teleport_button,
            self.creamy_tp_button,
            self.teleport_delay,
            self.save_point_button,
            self.open_storage_cog,
            self.hp_button,
            self.sp_button,
            self.sit_on_low_sp_button,
            self.sit_on_low_sp_toggle,
            self.fly_wings_amount,
            self._add_timer_button,
        ):
            widget.configure(state=state)
        for box in self._timer_boxes:
            key = box["key"]
            delay = box["delay"]
            remove = box["remove"]
            assert isinstance(key, ttk.Entry)
            assert isinstance(delay, ttk.Entry)
            assert isinstance(remove, ttk.Button)
            key.configure(state=state)
            delay.configure(state=state)
            remove.configure(state=state)
        if not locked:
            self._refresh_timer_add_button()

    # ══════════════════════════════════════════════════════════════════
    #  SHUTDOWN
    # ══════════════════════════════════════════════════════════════════

    def on_exit(self) -> None:
        """Clean shutdown of bot, VIIPER, hotkey, and session."""
        self._apply_ui_settings()
        if self.lifecycle.state != BotState.OFF:
            self.stop_bot()
        self.lifecycle.await_shutdown(timeout=DEFAULT_STOP_JOIN_TIMEOUT_S + 1.0)
        if self._memory_poll_after_id is not None:
            try:
                self.root.after_cancel(self._memory_poll_after_id)
            except tk.TclError:
                pass
            self._memory_poll_after_id = None
        if self._status_panel_poll_after_id is not None:
            try:
                self.root.after_cancel(self._status_panel_poll_after_id)
            except tk.TclError:
                pass
            self._status_panel_poll_after_id = None
        self._status_panel_overlay.destroy()
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
