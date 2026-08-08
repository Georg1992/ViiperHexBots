"""ViiperHexBots main window (tkinter) — UI building and callback wiring only.

Lifecycle logic             → :mod:`pybot.app.bot_lifecycle`
Hotkey registration/polling → :mod:`pybot.app.hotkey_manager`
Thread-safe log dispatch    → :mod:`pybot.app.log_pipe`
"""

from __future__ import annotations

import copy
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from pybot.app.bot_lifecycle import BotLifecycleManager, BotState
from pybot.app.config_store import AppConfig, list_client_profiles
from pybot.app.hotkey_manager import HotkeyManager
from pybot.app.log_pipe import LogPipe
from pybot.app.memory_stats_feed import MemoryStatsFeed
from pybot.app.overlay import StatusPanelOverlay, Win32HuntOverlay
from pybot.game_state import PlayerVitals
from pybot.app.session_log import AppSessionLog
from pybot.app.status_display import format_pair
from pybot.app.status_panel_feed import StatusPanelFeed
from pybot.app.ui_work_queue import UiWorkQueue
from pybot.app.startup_splash import preload_mob_descriptors
from pybot.app.viiper_manager import ViiperManager
from pybot.runtime.input.viiper_backend import ViiperStreamStore
from pybot.app.win32_util import (
    enum_game_windows,
    restore_and_activate,
    window_exists,
)
from pybot.mobs.import_mob import (
    MobImportError,
    import_mob_from_paths,
    mob_assets_exist,
    resolve_spr_act_paths,
)
from pybot.config.clients import memory_reading_enabled
from pybot.app.storage_chain_dialog import (
    StorageChainDialog,
    format_storage_chain_summary,
)
from pybot.app.mob_behavior_dialog import MobBehaviorDialog
from pybot.config.schema import (
    MAX_SKILL_TIMERS,
    KeyChainStep,
    MobCustomSettings,
    SkillTimerSetting,
)
from pybot.mobs.catalog import load_mob_catalog
from pybot.runtime.mob_behaviors import mob_has_custom_behavior
from pybot.runtime.input.scan_codes import keysym_to_key_name
from pybot.recognition.detector.detector import configure_opencv_runtime


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
        self.vitals = PlayerVitals()

        # ── Managers (created before UI so callbacks are ready) ─────
        # One process-wide VIIPER stream store is shared by the device
        # manager and every hunt input backend, so device streams survive
        # Stop/Start and are closed once on application exit.
        self.stream_store = ViiperStreamStore()
        self.log_pipe = LogPipe(self.root)
        self.viiper = ViiperManager(
            stream_store=self.stream_store,
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
            vitals=self.vitals,
            stream_store=self.stream_store,
            on_state_change=self._on_bot_state_changed,
            on_log=self.log_pipe.log,
            on_input_ready=self._enable_after_viiper,
            on_exit_requested=self.on_exit,
        )
        self.hotkey_manager = HotkeyManager(
            root=self.root,
            on_hotkey=self.toggle_bot,
        )

        # Background observation feeds (process-memory reads and status-panel
        # OCR) run off the Tk thread; results arrive via posted callbacks.
        self._memory_feed = MemoryStatsFeed(
            root=self.root,
            config=self.config,
            vitals=self.vitals,
            log=self.log_pipe.log,
            post_to_tk=self._post_ui_callback,
            on_name=self._set_memory_name,
            on_sp=self._set_memory_sp,
            on_weight=self._set_memory_weight,
        )
        self._status_feed = StatusPanelFeed(
            root=self.root,
            config=self.config,
            vitals=self.vitals,
            overlay=self._status_panel_overlay,
            log=self.log_pipe.log,
            post_to_tk=self._post_ui_callback,
            on_hp=self._set_memory_hp,
            on_sp=self._set_memory_sp,
            on_weight=self._set_memory_weight,
        )

        # ── UI state ────────────────────────────────────────────────

        self.window_entries: list = []
        self.mob_var = tk.IntVar(
            value=min(max(1, self.config.selected_monster), max(1, len(self.mob_catalog)))
            if self.mob_catalog
            else 1
        )
        # Unbounded put_nowait is intentional: a worker completion callback must
        # never be dropped while its pending flag is set.
        self._ui_callback_queue: queue.Queue[callable] = queue.Queue()
        self._exit_requested = False
        self._config_work = UiWorkQueue(name="ui-config-writer")
        self._shutdown_work = UiWorkQueue(name="ui-shutdown")
        self._shutdown_cleanup_pending = False
        self._resume_work = UiWorkQueue(name="ui-resume")
        self._resume_pending = False
        # Ignore widget callbacks while building; enable at end of _build_ui.
        self._settings_apply_enabled = False
        self._mob_radios: list[ttk.Radiobutton] = []
        self._mob_settings_buttons: list[ttk.Button] = []
        self._settings_checkbuttons: list[ttk.Checkbutton] = []
        self._mob_import_busy = False
        self._mob_radio_frame: ttk.Frame | None = None

        # Build UI (widgets created here, references shared to managers)
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        self._fit_window_to_content()

        # Wire the non-log status widget and durable session sink after widgets exist.
        # Log messages intentionally have no Tk or in-game visual sink.
        self.log_pipe.set_status_widgets(self.input_status, self.input_hint)
        self.log_pipe.set_persist_callback(self._persist_log_line)

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
        for button in self._mob_settings_buttons:
            button.destroy()
        self._mob_radios.clear()
        self._mob_settings_buttons.clear()
        for index, mob in enumerate(self.mob_catalog, start=1):
            label = mob.display_name
            if mob_has_custom_behavior(mob.descriptor_name):
                label = f"{mob.display_name}  (legacy custom)"
            radio = ttk.Radiobutton(
                frame,
                text=label,
                variable=self.mob_var,
                value=index,
                command=self._apply_ui_settings,
            )
            radio.grid(row=index - 1, column=0, sticky="w")
            self._mob_radios.append(radio)
            settings_button = ttk.Button(
                frame,
                text="⚙",
                width=3,
                command=lambda name=mob.descriptor_name: self._open_mob_behavior_dialog(name),
            )
            settings_button.grid(row=index - 1, column=1, sticky="w", padx=(6, 0))
            self._mob_settings_buttons.append(settings_button)
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

    def _post_ui_callback(self, callback) -> None:
        """Queue a UI callback without calling Tk from a worker thread."""
        self._ui_callback_queue.put_nowait(callback)

    def _drain_ui_callbacks(self) -> None:
        processed = 0
        try:
            while processed < 20:
                callback = self._ui_callback_queue.get_nowait()
                try:
                    callback()
                except Exception as exc:
                    # One failed widget/overlay update must not strand later
                    # feed results or prevent the next drain tick.
                    self.log_pipe.log(f"[UI] callback error: {exc}")
                processed += 1
        except queue.Empty:
            pass
        finally:
            try:
                if self.root.winfo_exists():
                    self.root.after(50, self._drain_ui_callbacks)
            except tk.TclError:
                pass

    def _save_config_async(self) -> None:
        """Persist a snapshot off the Tk thread; never block a widget callback."""
        snapshot = copy.deepcopy(self.config)

        def _save() -> None:
            try:
                snapshot.save()
            except Exception as exc:
                self._post_ui_callback(
                    lambda exc=exc: self.log_pipe.log(
                        f"[UI] Settings save failed: {exc}"
                    )
                )

        self._config_work.submit(_save)

    def _browse_mob_assets(self) -> None:
        if not self._can_import_mob():
            return
        paths = filedialog.askopenfilenames(
            title="Select exactly one .spr and one .act file (both with same name)",
            filetypes=[
                ("SPR/ACT", "*.spr *.act"),
                ("SPR", "*.spr"),
                ("ACT", "*.act"),
                ("All", "*.*"),
            ],
        )
        if paths:
            self._begin_mob_import([Path(p) for p in paths])

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
            self._mob_import_status.configure(text=str(exc))
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
        self._mob_import_status.configure(text=f"Building {stem}…")
        self.log_pipe.log(f"[MOB] importing {spr.name} + {act.name}")

        def _worker() -> None:
            try:
                entry = import_mob_from_paths([spr, act], overwrite=overwrite)
            except Exception as exc:
                err = exc
                self._post_ui_callback(lambda err=err: self._mob_import_failed(err))
                return
            stem_ready = entry.descriptor_name
            self._post_ui_callback(
                lambda stem_ready=stem_ready: self._mob_import_succeeded(stem_ready)
            )

        threading.Thread(target=_worker, name="mob-import", daemon=True).start()

    def _mob_import_failed(self, exc: Exception) -> None:
        self._mob_import_busy = False
        if self.lifecycle.state == BotState.OFF:
            self._mob_browse_button.configure(state=tk.NORMAL)
        self._mob_import_status.configure(text=f"Failed: {exc}")
        self.log_pipe.log(f"[MOB] import failed: {exc}")
        messagebox.showerror("Import mob", f"Failed to build descriptor:\n\n{exc}")

    def _mob_import_succeeded(self, stem: str) -> None:
        self._mob_import_busy = False
        if self.lifecycle.state == BotState.OFF:
            self._mob_browse_button.configure(state=tk.NORMAL)
        self._refresh_mob_radios(select_stem=stem)
        self._mob_import_status.configure(text=f"Ready: {stem}")
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

        import_frame = ttk.LabelFrame(mob_col, text="Add mob (SPR + ACT)", padding=6)
        import_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self._mob_browse_label = tk.Label(
            import_frame,
            text="Select one .spr + one .act file\n(both must have the same name)",
            relief=tk.GROOVE,
            borderwidth=2,
            width=28,
            height=3,
            justify=tk.CENTER,
            background="#f0f0f0",
        )
        self._mob_browse_label.grid(row=0, column=0, sticky="ew")
        browse_btns = ttk.Frame(import_frame)
        browse_btns.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._mob_browse_button = ttk.Button(
            browse_btns,
            text="Browse…",
            command=self._browse_mob_assets,
        )
        self._mob_browse_button.pack(side=tk.LEFT)
        self._mob_import_status = ttk.Label(import_frame, text="", wraplength=200)
        self._mob_import_status.grid(row=2, column=0, sticky="w", pady=(4, 0))

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

        self.use_sprite_grf_var = tk.BooleanVar(value=self.config.use_sprite_grf)
        sprite_check = ttk.Checkbutton(
            mode_col,
            text="Use sprite.grf",
            variable=self.use_sprite_grf_var,
            command=self._apply_ui_settings,
        )
        sprite_check.grid(row=3, column=0, sticky="w", pady=(6, 0))
        self._settings_checkbuttons.append(sprite_check)

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
        ttk.Label(hp_cell, text="HP Item Key:").pack(side=tk.LEFT)
        self.hp_button = ttk.Entry(hp_cell, width=6)
        self.hp_button.insert(0, self.config.hp_button)
        self.hp_button.pack(side=tk.LEFT, padx=(4, 0))
        self._bind_key_capture(self.hp_button)
        self._bind_setting_entry(self.hp_button)
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
        # 49 = Off (AHK); 50–85 = active threshold %.
        initial_weight = max(49, min(85, int(self.config.weight_modifier)))
        self.storage_weight = tk.IntVar(value=initial_weight)
        self.storage_weight_scale = ttk.Scale(
            weight_cell,
            from_=49,
            to=85,
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

        # ── Controls ────────────────────────────────────────────────
        controls = ttk.Frame(main)
        controls.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 0))
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
        self._sync_memory_reading_from_profile()
        self._update_search_label()
        self._memory_feed.set_active(False)
        self._status_feed.set_active(False)
        self._memory_feed.start()
        self._status_feed.start()
        # Observation feeds are intentionally idle until the bot owns a live
        # session. The labels remain visible, but stopped/paused states must
        # not read process memory or OCR the game window.
        self.root.after(50, self._drain_ui_callbacks)
        self._settings_apply_enabled = True

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

    def _open_mob_behavior_dialog(self, mob_name: str) -> None:
        key = mob_name.strip().lower()
        current = self.config.mob_custom_settings.get(key, MobCustomSettings())

        def _apply(settings: MobCustomSettings) -> None:
            self.config.mob_custom_settings[key] = settings
            self._save_config_async()
            self.log_pipe.log(f"[MOB] custom behavior saved: {mob_name}")

        MobBehaviorDialog(
            self.root,
            mob_name,
            current,
            on_apply=_apply,
        )

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
        # A new window invalidates any in-flight read; restart both feeds so
        # the next submit targets the freshly selected window.
        self._memory_feed.reset()
        self._memory_feed.request_now()
        self._status_feed.reset()
        self._status_feed.request_now()
        if self._settings_apply_enabled:
            self._save_config_async()

    def on_client_changed(self, *_event) -> None:
        self.config.client_profile = self.client_combo.get()
        self._sync_memory_reading_from_profile()
        self._memory_feed.reset()
        self._memory_feed.request_now()
        self._status_feed.reset()
        self._status_feed.request_now()
        memory = "on" if self.config.use_memory_reading else "off"
        if self.config.use_memory_reading:
            source = "memory (HP from status panel)"
        else:
            source = "status panel"
        self.log_pipe.log(
            f"Client profile: {self.config.client_profile} "
            f"(memory reading {memory}, stats from {source})"
        )
        if self._settings_apply_enabled:
            self._save_config_async()

    def _sync_memory_reading_from_profile(self) -> None:
        """Memory reading follows the profile: Generic off, server profiles on."""
        self.config.use_memory_reading = memory_reading_enabled(self.client_combo.get())

    @staticmethod
    def _format_pair(current: int | None, maximum: int | None) -> str:
        """Compatibility wrapper around the pure status display helper."""
        return format_pair(current, maximum)

    def _set_memory_name(self, text: str) -> None:
        self.memory_name.configure(text=text)

    def _set_memory_hp(self, text: str) -> None:
        self.memory_hp.configure(text=text)

    def _set_memory_sp(self, text: str) -> None:
        self.memory_sp.configure(text=text)

    def _set_memory_weight(self, text: str) -> None:
        self.memory_weight.configure(text=text)

    def _persist_log_line(self, message: str) -> None:
        """LogPipe sink — mirror every UI log line into the session log.

        Registered in ``__init__`` via ``set_persist_callback``; runs on the
        Tk main thread during drain. write_system only enqueues
        (non-blocking) and silently drops before a session opens or after it
        closes, so this is safe and cheap.
        """
        self.session.write_system("INFO", "ui", message)



    def _sync_config_from_ui(self) -> None:
        """Read all UI widget values into self.config."""
        self.config.client_profile = self.client_combo.get()
        self._sync_memory_reading_from_profile()
        self.config.selected_monster = self.mob_var.get()
        self.config.hunt_mode = self.hunt_mode_var.get()
        self.config.search_range = int(float(self.search_range.get()))
        self.config.take_fly_wings = self.fly_wings_var.get()
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
        self.config.sp_button = self.sp_button.get().strip()
        self.config.sit_on_low_sp_button = self.sit_on_low_sp_button.get().strip()
        self.config.sit_on_low_sp = self.sit_on_low_sp_var.get()
        self.config.use_sprite_grf = self.use_sprite_grf_var.get()
        raw = self.fly_wings_amount.get().strip()
        self.config.fly_wings_amount = int(raw) if raw else 0

    def _apply_ui_settings(self, *_args) -> None:
        """Push current GUI values into config.ini as soon as they change."""
        if not self._settings_apply_enabled:
            return
        try:
            self._sync_config_from_ui()
            self._save_config_async()
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
        elif self.lifecycle.state == BotState.STOPPING or self.lifecycle.stopping:
            self.log_pipe.log("Stop already in progress; waiting for workers to exit")
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
            self._save_config_async()
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

    def resume_bot(self) -> None:
        """Restore focus asynchronously, then resume on the Tk thread."""
        if self._resume_pending:
            return
        self._resume_pending = True
        hwnd = self.config.window_id

        def _restore() -> None:
            try:
                restore_and_activate(hwnd)
            except Exception as exc:
                self._post_ui_callback(lambda exc=exc: self._finish_resume(exc))
                return
            self._post_ui_callback(self._finish_resume)

        if not self._resume_work.submit(_restore):
            self._resume_pending = False

    def _finish_resume(self, error: Exception | None = None) -> None:
        self._resume_pending = False
        if error is not None:
            self.log_pipe.log(f"[UI] Could not restore game focus: {error}")
            return
        if self.lifecycle.resume():
            self.log_pipe.log("Bot resumed")
        else:
            self.log_pipe.log(
                "Bot resume deferred — input is still unwinding; try Continue again"
            )

    # ══════════════════════════════════════════════════════════════════
    #  CALLBACKS (registered with lifecycle and log pipe)
    # ══════════════════════════════════════════════════════════════════

    def _enable_after_viiper(self) -> None:
        """Enable UI widgets after VIIPER is ready (runs on main thread)."""
        self.window_combo.configure(state="readonly")
        self.bot_button.configure(state=tk.NORMAL)
        self.log_pipe.log("All set — select or launch the game window")
        self.refresh_windows()

    def _set_observation_feeds_active(self, active: bool) -> None:
        """Allow UI observation only while the bot is starting/running.

        ``set_active`` invalidates in-flight generations, preventing a native
        memory/OCR completion from updating vitals after pause or stop.
        """
        self._memory_feed.set_active(active)
        self._status_feed.set_active(active)

    def _on_bot_state_changed(self, state: BotState) -> None:
        """Update UI widgets to reflect the current bot state."""
        self._set_observation_feeds_active(
            state in (BotState.STARTING, BotState.RUNNING)
        )
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
        elif state == BotState.STOPPING:
            self.bot_button.configure(text="Stopping...", state=tk.DISABLED)
            self.continue_button.configure(state=tk.DISABLED)
            self.bot_status.configure(text="Stopping...")
            self.status_indicator.configure(text=" STOP ", bg="#6d4c41")
            self._lock_ui(True)
        elif state == BotState.PAUSED:
            self.bot_button.configure(text="Stop Bot")
            self.bot_status.configure(text="Paused (game focus lost)")
            self.status_indicator.configure(text=" PAUSED ", bg="#f9a825")
            self.continue_button.configure(state=tk.NORMAL)
        elif state == BotState.OFF:
            self.bot_button.configure(text="Start Bot", state=tk.NORMAL)
            self.continue_button.configure(state=tk.DISABLED)
            self.bot_status.configure(text="Off")
            self.status_indicator.configure(text="  OFF  ", bg="#c62828")
            self._lock_ui(False)

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
        for button in self._mob_settings_buttons:
            button.configure(state=state)
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
        """Begin shutdown without joining or tearing down services on Tk."""
        if self._exit_requested:
            return
        self._exit_requested = True
        self._apply_ui_settings()
        if (
            self.lifecycle.state != BotState.OFF
            or not self.lifecycle.shutdown_ready
        ):
            self.stop_bot()
        self.bot_button.configure(state=tk.DISABLED, text="Closing...")
        self.log_pipe.log("Closing bot and stopping VIIPER...")
        self._poll_shutdown()

    def _poll_shutdown(self) -> None:
        """Keep Tk responsive while lifecycle workers unwind."""
        self.lifecycle.retry_shutdown_cleanup()
        if not self.lifecycle.shutdown_ready:
            self.root.after(50, self._poll_shutdown)
            return
        if not self._shutdown_cleanup_pending:
            self._shutdown_cleanup_pending = True

            def _cleanup() -> None:
                try:
                    self.viiper.shutdown()
                    self.session.end("user exit")
                except Exception as exc:
                    self._post_ui_callback(
                        lambda exc=exc: self.log_pipe.log(
                            f"[UI] Shutdown cleanup failed: {exc}"
                        )
                    )
                finally:
                    self._post_ui_callback(self._finish_exit)

            if not self._shutdown_work.submit(_cleanup):
                self._shutdown_cleanup_pending = False
                self.root.after(50, self._poll_shutdown)
            return
        # The worker posts _finish_exit; this branch only keeps the callback
        # chain alive if a custom queue implementation delays delivery.
        self.root.after(50, self._poll_shutdown)

    def _finish_exit(self) -> None:
        """Close auxiliary workers, then destroy Tk after they drain."""
        if not self._exit_requested:
            return
        self._status_panel_overlay.destroy()
        self.hotkey_manager.destroy()
        self._memory_feed.close()
        self._status_feed.close()
        self._config_work.close()
        self._shutdown_work.close()
        self._resume_work.close()
        self._poll_auxiliary_shutdown()

    def _poll_auxiliary_shutdown(self) -> None:
        """Wait for accepted UI work without blocking Tk."""
        queues = (
            self._memory_feed,
            self._status_feed,
            self._config_work,
            self._shutdown_work,
            self._resume_work,
        )
        if not all(work.idle for work in queues):
            self.root.after(25, self._poll_auxiliary_shutdown)
            return
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    # Configure OpenCV before preload or the autonomous status OCR reader can
    # start. The GUI feed and hunt observers share this process, and leaving
    # OpenCV's default native pool active during startup lets tiny OCR calls
    # contend with discovery/tracking for the same CPU/native resources.
    configure_opencv_runtime()
    # Build/refresh descriptors before the main GUI so hunt never races a rebuild.
    if not preload_mob_descriptors():
        return
    MainWindow().run()


if __name__ == "__main__":
    main()
