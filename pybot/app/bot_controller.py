"""In-process hunt runtime controller."""

from __future__ import annotations

import threading
from collections.abc import Callable

from pybot.app.config_store import AppConfig
from pybot.app.mob_catalog import load_mob_catalog, mob_folder_by_index
from pybot.paths import PROJECT_ROOT
from pybot.runtime.config import load_runtime_config
from pybot.runtime.hunt_runtime import create_runtime_deps, HuntRuntime


class BotController:
    def __init__(
        self,
        *,
        app_config: AppConfig,
        session_id: str,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        self._app_config = app_config
        self._session_id = session_id
        self._on_log = on_log
        self._runtime: HuntRuntime | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return

        catalog = load_mob_catalog()
        mob_name = mob_folder_by_index(catalog, self._app_config.selected_monster)
        control_file = PROJECT_ROOT / "logs" / "sessions" / self._session_id / "control.json"
        runtime_config = load_runtime_config(
            config_path=self._app_config.config_path,
            hwnd=self._app_config.window_id,
            mob_name=mob_name,
            validation_enabled=self._app_config.hunt_validation_log,
            control_file=control_file,
            session_id=self._session_id,
        )
        deps = create_runtime_deps(
            runtime_config,
            session_id=self._session_id,
            behavior_callback=self._on_log,
        )
        self._runtime = HuntRuntime(deps)
        self._thread = threading.Thread(
            target=self._runtime.run,
            name="hunt-runtime",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._runtime is not None:
            self._runtime.stop()
        if self._thread is not None:
            self._thread.join(timeout=0.1)
        self._thread = None
        self._runtime = None

    def pause(self) -> None:
        if self._runtime is not None:
            self._runtime.pause()

    def resume(self) -> None:
        if self._runtime is not None:
            self._runtime.resume()
