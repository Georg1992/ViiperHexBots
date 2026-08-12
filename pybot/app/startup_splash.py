"""Pre-GUI splash while mob descriptors are checked / built."""

from __future__ import annotations

import queue
import threading
from tkinter import messagebox

from pybot.app.loading_dialog import LoadingDialog
from pybot.mobs.catalog import ensure_mob_assets, load_mob_catalog
from pybot.recognition.detector.descriptors.descriptor_builder import DESCRIPTOR_VERSION

_SENTINEL = object()


class DescriptorLoadingSplash:
    """Modal splash that blocks until ensure_mob_assets finishes."""

    def __init__(self) -> None:
        self._loading = LoadingDialog(
            title="ViiperHex Bot",
            heading="ViiperHex Bot",
            message=f"Preparing mob descriptors (v{DESCRIPTOR_VERSION})…",
        )
        self.root = self._loading.window
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_attempt)

        self._messages: queue.Queue[object] = queue.Queue()
        self._failed = False
        self._closed_early = False
        self._worker_done = False

    def _on_close_attempt(self) -> None:
        if self._worker_done:
            self._loading.close()
            return
        self._closed_early = True
        self._failed = True
        self._loading.close()

    def _log(self, message: str) -> None:
        self._messages.put(message)

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self._messages.get_nowait()
                if item is _SENTINEL:
                    self._worker_done = True
                    self._finish()
                    return
                self._loading.set_message(str(item))
        except queue.Empty:
            pass
        if not self._closed_early:
            self.root.after(50, self._poll_queue)

    def _finish(self) -> None:
        self._loading.stop()
        catalog = load_mob_catalog(ensure_assets=False)
        if not catalog:
            self._failed = True
            messagebox.showerror(
                "ViiperHex Bot",
                "No mob descriptors available after startup build.\n\n"
                "Check assets/mobs for SPR/ACT pairs and the log above for "
                "[AUTO-BUILD] errors.",
                parent=self.root,
            )
        self._loading.close()

    def _worker(self) -> None:
        try:
            ensure_mob_assets(log_fn=self._log)
        except Exception as exc:
            self._log(f"[AUTO-BUILD] startup failed — {exc}")
            self._failed = True
        finally:
            self._messages.put(_SENTINEL)

    def run(self) -> bool:
        """Run splash + asset prep. Returns True when the main GUI may open."""
        threading.Thread(
            target=self._worker,
            daemon=True,
            name="descriptor-preload",
        ).start()
        self.root.after(50, self._poll_queue)
        self.root.mainloop()
        return not self._failed and not self._closed_early


def preload_mob_descriptors() -> bool:
    """Show loading UI, ensure descriptors, then return readiness for MainWindow."""
    return DescriptorLoadingSplash().run()
