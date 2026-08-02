"""In-process hunt runtime controller."""

from __future__ import annotations

import threading
from collections.abc import Callable

from pybot.app.config_store import AppConfig
from pybot.config.runtime import load_runtime_config
from pybot.game_state import PlayerVitals
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
        vitals: PlayerVitals | None = None,
    ) -> None:
        self._app_config = app_config
        self._session_id = session_id
        self._on_log = on_log
        self._overlay = NullOverlay() if overlay is None else overlay
        self._vitals = PlayerVitals() if vitals is None else vitals
        self._runtime: HuntRuntime | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def shutdown_pending(self) -> bool:
        """True while this controller still owns a live runtime or worker set."""
        if self._thread is not None and self._thread.is_alive():
            return True
        runtime = self._runtime
        complete = getattr(runtime, "is_shutdown_complete", None)
        if not callable(complete):
            # Lightweight/custom runtimes predate the completion handshake;
            # once their top-level thread exits, preserve old compatibility.
            return False
        return complete() is False

    def start(self, *, mob_name: str) -> None:
        if self.shutdown_pending:
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
            vitals=self._vitals,
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
        """Stop the hunt and release ownership only after full cleanup.

        The top-level runtime thread can return after its first bounded worker
        shutdown attempt while a non-cooperative worker is still alive. In
        that case the runtime object remains owned and a later stop retries
        cleanup; a new hunt can never be started over those workers.
        """
        self.request_stop()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                return False

        runtime = self._runtime
        complete = getattr(runtime, "is_shutdown_complete", None)
        if callable(complete) and complete() is False:
            retry = getattr(runtime, "retry_shutdown", None)
            if not callable(retry) or retry() is not True:
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

    def resume(self) -> bool:
        if self._runtime is None:
            return False
        result = self._runtime.resume()
        return result is not False

    def set_search_range_cells(self, cells: int) -> None:
        if self._runtime is not None:
            self._runtime.set_search_range_cells(cells)
