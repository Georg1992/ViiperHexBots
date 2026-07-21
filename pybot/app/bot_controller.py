"""In-process hunt runtime controller."""

from __future__ import annotations

import threading
from collections.abc import Callable

from pybot.app.config_store import AppConfig
from pybot.config.runtime import load_runtime_config
from pybot.paths import SESSIONS_DIR
from pybot.runtime.hunt_runtime import create_runtime_deps, HuntRuntime
from pybot.runtime.overlay_ports import HuntOverlay, NullOverlay

DEFAULT_STOP_JOIN_TIMEOUT_S = 3.0


class BotController:
    def __init__(
        self,
        *,
        app_config: AppConfig,
        session_id: str,
        on_log: Callable[[str], None] | None = None,
        overlay: HuntOverlay | None = None,
    ) -> None:
        self._app_config = app_config
        self._session_id = session_id
        self._on_log = on_log
        self._overlay = overlay or NullOverlay()
        self._runtime: HuntRuntime | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, *, mob_name: str) -> None:
        if self.running:
            return

        control_file = SESSIONS_DIR / self._session_id / "control.json"
        runtime_config = load_runtime_config(
            settings=self._app_config,
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
            overlay=self._overlay,
        )
        self._runtime = HuntRuntime(deps)
        self._thread = threading.Thread(
            target=self._runtime.run,
            name="hunt-runtime",
            daemon=True,
        )
        self._thread.start()

    def request_stop(self) -> None:
        """Signal the hunt runtime to stop without blocking."""
        if self._runtime is not None:
            self._runtime.stop()

    def stop(self, *, join_timeout: float = DEFAULT_STOP_JOIN_TIMEOUT_S) -> bool:
        """Stop the hunt runtime and join its thread.

        Returns True when the hunt thread has exited. If the join times out the
        controller keeps its handles so ``running`` stays True and a later
        ``stop`` / start-await can finish the shutdown — never overlap two hunts.
        """
        self.request_stop()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                return False
        self._thread = None
        self._runtime = None
        control_file = SESSIONS_DIR / self._session_id / "control.json"
        if control_file.is_file():
            control_file.unlink(missing_ok=True)
        return True

    def pause(self) -> None:
        if self._runtime is not None:
            self._runtime.pause()

    def resume(self) -> None:
        if self._runtime is not None:
            self._runtime.resume()

    def set_search_range_cells(self, cells: int) -> None:
        if self._runtime is not None:
            self._runtime.set_search_range_cells(cells)
