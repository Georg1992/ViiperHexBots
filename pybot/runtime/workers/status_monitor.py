"""Periodically feed vitals from the shared store into danger/tracking consumers.

Runs in the hunt runtime alongside attack/tracking workers.  The UI publishes
HP/SP/Weight from status-panel OCR or process memory into ``PlayerVitals``.
This worker polls that store and feeds the ``DangerDetector`` with HP so it
can detect critical HP drops during normal hunting.
"""

from __future__ import annotations

import time

from pybot.runtime.constants import WORKER_POLL_INTERVAL_S


class StatusMonitor:
    """Poll ``PlayerVitals`` and feed consumers (danger, etc.)."""

    def __init__(
        self,
        ctx,
        vitals,
        danger,
        *,
        poll_interval_s: float = WORKER_POLL_INTERVAL_S,
    ) -> None:
        self._ctx = ctx
        self._vitals = vitals
        self._danger = danger
        self._poll_interval_s = poll_interval_s

    def run(self) -> None:
        """Ongoing loop: read vitals, feed consumers, sleep."""
        while not self._ctx.is_stopped():
            hp, hp_max = self._vitals.hp_pair()
            if hp is not None:
                self._danger.feed_hp(hp, hp_max)
            self._ctx.stop_event.wait(self._poll_interval_s)
