"""Process-memory observation feed for the main window.

Periodically reads the client's SP/weight from process memory and publishes
the values into the shared :class:`PlayerVitals` plus the profile labels.
The Win32 read runs on the internal worker thread; label/vitals updates
happen on the Tk thread via the runner's result callback.
"""

from __future__ import annotations

from collections.abc import Callable

from pybot.app.periodic_task_runner import PeriodicTaskRunner
from pybot.app.status_display import format_pair
from pybot.app.win32_util import window_exists
from pybot.config.clients import load_client_profile
from pybot.game_state import GameMemoryPoller

MEMORY_POLL_MS = 500
# A memory read must never pin its pending flag forever: a wedged read would
# silently kill the SP/weight feed (and with it storage/SP automation) with
# no error and no log line. After this long the worker is abandoned and
# recreated.
MEMORY_READ_TIMEOUT_S = 6.0


class MemoryStatsFeed(PeriodicTaskRunner):
    """One observation feed: periodic process-memory SP/weight reads."""

    def __init__(
        self,
        *,
        root,
        config,
        vitals,
        log: Callable[[str], None],
        post_to_tk: Callable[[Callable[[], None]], None],
        on_name: Callable[[str], None],
        on_sp: Callable[[str], None],
        on_weight: Callable[[str], None],
    ) -> None:
        super().__init__(
            root=root,
            name="ui-memory-reader",
            timeout_s=MEMORY_READ_TIMEOUT_S,
            default_delay_ms=MEMORY_POLL_MS,
            post_to_tk=post_to_tk,
            log=log,
        )
        self._config = config
        self._vitals = vitals
        self._poller = GameMemoryPoller()
        self._on_name = on_name
        self._on_sp = on_sp
        self._on_weight = on_weight

    def reset(self) -> None:
        super().reset()
        self._poller.reset()

    def should_submit(self) -> int | None:
        if not self._config.use_memory_reading:
            self._on_name("—")
            return None
        profile = load_client_profile(self._config.client_profile)
        if profile is None or not profile.memory.has_any:
            self._on_name("—")
            self._on_sp("—")
            self._on_weight("—")
            self._vitals.clear_sp()
            return None
        hwnd = self._config.window_id
        if not hwnd or not window_exists(hwnd):
            self._on_name("—")
            self._on_sp("—")
            self._on_weight("—")
            self._vitals.clear_sp()
            return None
        return MEMORY_POLL_MS

    def build_job(self, generation: int) -> Callable[[], None] | None:
        hwnd = self._config.window_id
        profile = load_client_profile(self._config.client_profile)
        if profile is None or not profile.memory.has_any:
            return None

        def _read() -> None:
            try:
                snap = self._poller.read(hwnd, profile.memory)
            except Exception as exc:
                self.fail(generation, exc)
                return
            self.publish(generation, (hwnd, snap))

        return _read

    def apply_result(self, result) -> None:
        hwnd, snap = result
        if hwnd != self._config.window_id or not self._config.use_memory_reading:
            return
        if not snap.ok:
            self._on_name("—")
            self._on_sp("—")
            self._on_weight("—")
            self._vitals.clear_sp()
            return
        self._on_name(snap.char_name or "—")
        self._on_sp(format_pair(snap.sp, snap.sp_max))
        self._on_weight(format_pair(snap.weight, snap.weight_max))
        self._vitals.publish_sp(snap.sp, snap.sp_max)
        self._vitals.publish_weight(snap.weight, snap.weight_max)
        # HP is vision-only — never overwrite from memory polls.

    def on_failure(self, exc: Exception, generation: int) -> None:
        self._log(f"[UI] Memory read failed: {exc}")

    def on_recover(self, stall_count: int) -> None:
        self._log(
            "[UI] Memory read stalled — restarted memory reader "
            f"(stall #{stall_count})"
        )
