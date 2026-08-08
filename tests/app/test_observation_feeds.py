"""Unit tests for the extracted observation feeds (no Tk root required)."""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pybot.app.memory_stats_feed import MEMORY_POLL_MS, MemoryStatsFeed
from pybot.app.periodic_task_runner import PeriodicTaskRunner
from pybot.app.status_panel_feed import StatusPanelFeed


class _FakeVitals:
    def __init__(self) -> None:
        self.sp = None
        self.weight = None
        self.hp = None
        self.clear_count = 0

    def clear_sp(self) -> None:
        self.clear_count += 1
        self.sp = None
        self.weight = None

    def publish_sp(self, value, maximum) -> None:
        self.sp = (value, maximum)

    def publish_weight(self, value, maximum) -> None:
        self.weight = (value, maximum)

    def publish_hp(self, value, maximum) -> None:
        self.hp = (value, maximum)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, *args) -> None:
        self.calls.append(args)


def _memory_feed(**overrides) -> MemoryStatsFeed:
    labels = {name: _Recorder() for name in ("name", "sp", "weight")}
    feed = MemoryStatsFeed(
        root=object(),
        config=SimpleNamespace(
            use_memory_reading=True,
            client_profile="Generic",
            window_id=123,
        ),
        vitals=_FakeVitals(),
        log=lambda _msg: None,
        post_to_tk=lambda cb: None,
        on_name=labels["name"],
        on_sp=labels["sp"],
        on_weight=labels["weight"],
    )
    feed._poller = MagicMock()
    for key, value in overrides.items():
        setattr(feed._config, key, value)
    return feed


class MemoryStatsFeedTests(unittest.TestCase):
    def test_skips_when_memory_reading_disabled(self) -> None:
        feed = _memory_feed(use_memory_reading=False)
        with patch("pybot.app.memory_stats_feed.load_client_profile") as load:
            delay = feed.should_submit()
        self.assertIsNone(delay)
        load.assert_not_called()
        self.assertEqual(feed._on_name.calls, [("—",)])

    @patch("pybot.app.memory_stats_feed.window_exists", return_value=True)
    @patch(
        "pybot.app.memory_stats_feed.load_client_profile",
        return_value=SimpleNamespace(memory=SimpleNamespace(has_any=True)),
    )
    def test_submits_when_profile_and_window_ok(self, _load, _window) -> None:
        feed = _memory_feed()
        self.assertEqual(feed.should_submit(), MEMORY_POLL_MS)

    @patch(
        "pybot.app.memory_stats_feed.load_client_profile",
        return_value=SimpleNamespace(memory=SimpleNamespace(has_any=False)),
    )
    def test_clears_labels_without_profile_addresses(self, _load) -> None:
        feed = _memory_feed()
        feed._vitals.clear_sp()
        self.assertIsNone(feed.should_submit())
        self.assertEqual(feed._on_name.calls, [("—",)])
        self.assertEqual(feed._on_sp.calls, [("—",)])
        self.assertEqual(feed._on_weight.calls, [("—",)])
        self.assertEqual(feed._vitals.clear_count, 2)

    def test_applies_snapshot_to_labels_and_vitals(self) -> None:
        feed = _memory_feed()
        snap = SimpleNamespace(
            ok=True,
            char_name="Hero",
            sp=80,
            sp_max=100,
            weight=40,
            weight_max=50,
        )
        feed.apply_result((123, snap))
        self.assertEqual(feed._on_name.calls, [("Hero",)])
        self.assertEqual(feed._on_sp.calls, [("80/100",)])
        self.assertEqual(feed._on_weight.calls, [("40/50",)])
        self.assertEqual(feed._vitals.sp, (80, 100))
        self.assertEqual(feed._vitals.weight, (40, 50))

    def test_ignores_result_for_another_window(self) -> None:
        feed = _memory_feed()
        snap = SimpleNamespace(ok=True, char_name="Hero", sp=1, sp_max=2,
                               weight=3, weight_max=4)
        feed.apply_result((999, snap))
        self.assertEqual(feed._on_name.calls, [])
        self.assertEqual(feed._vitals.sp, None)

    def test_memory_reads_stop_when_feed_is_inactive(self) -> None:
        feed = _memory_feed()
        feed.set_active(False)
        with patch("pybot.app.memory_stats_feed.load_client_profile") as load:
            self.assertIsNone(feed.should_submit())
        load.assert_not_called()

    def test_inactive_memory_feed_drops_completed_result(self) -> None:
        feed = _memory_feed()
        feed.set_active(False)
        snap = SimpleNamespace(ok=True, char_name="Hero", sp=1, sp_max=2,
                               weight=3, weight_max=4)
        feed.apply_result((123, snap))
        self.assertEqual(feed._vitals.sp, None)


def _status_feed(**overrides) -> StatusPanelFeed:
    labels = {name: _Recorder() for name in ("hp", "sp", "weight")}
    feed = StatusPanelFeed(
        root=object(),
        config=SimpleNamespace(window_id=123, use_memory_reading=False),
        vitals=_FakeVitals(),
        overlay=MagicMock(),
        log=lambda _msg: None,
        post_to_tk=lambda cb: None,
        on_hp=labels["hp"],
        on_sp=labels["sp"],
        on_weight=labels["weight"],
    )
    for key, value in overrides.items():
        setattr(feed._config, key, value)
    return feed


class StatusPanelFeedTests(unittest.TestCase):
    def test_reader_publishes_sp_when_other_rows_fail(self) -> None:
        """SP recovery must not depend on HP/weight UI projection."""
        feed = _status_feed()
        result = SimpleNamespace(
            hwnd=123,
            state="sp_only",
            values=None,
            sp=(98, 100),
        )
        feed._record_reader_result(result, feed._lifecycle_epoch)
        self.assertEqual(feed._vitals.sp, (98, 100))

    def test_autonomous_reader_publishes_without_tk_callback(self) -> None:
        """OCR/vitals continue even when Tk has not drained presentation."""
        feed = _status_feed()
        reads = 0
        read_started = threading.Event()
        callbacks: list = []
        values = SimpleNamespace(
            panel_origin=(1, 2),
            hp=90, hp_max=100, sp=70, sp_max=100,
            weight=30, weight_max=50,
        )

        def read_snapshot():
            nonlocal reads
            reads += 1
            read_started.set()
            return SimpleNamespace(
                hwnd=123,
                state="values",
                values=values,
                client_left=10,
                client_top=20,
                client_width=300,
                client_height=200,
                full_refresh=True,
            )

        feed._read_snapshot = read_snapshot
        feed._post_to_tk = callbacks.append
        try:
            feed.start()
            self.assertTrue(read_started.wait(timeout=1.0))
            self.assertEqual(feed._vitals.sp, (70, 100))
            self.assertGreaterEqual(reads, 1)
            self.assertEqual(feed._on_sp.calls, [])
            self.assertEqual(len(callbacks), 1)
        finally:
            feed.close()
            self.assertTrue(self._wait_until(lambda: feed.idle))

    def test_reset_discards_queued_projection_from_previous_epoch(self) -> None:
        feed = _status_feed()
        callbacks: list = []
        feed._post_to_tk = callbacks.append
        values = SimpleNamespace(
            panel_origin=(1, 2),
            hp=90, hp_max=100, sp=70, sp_max=50,
            weight=30, weight_max=50,
        )
        result = SimpleNamespace(
            hwnd=123, state="values", values=values,
            client_left=10, client_top=20, full_refresh=True,
        )
        feed._record_reader_result(result, feed._lifecycle_epoch)
        feed._queue_ui_result(feed._lifecycle_epoch, result)
        self.assertEqual(len(callbacks), 1)
        feed.reset()
        callbacks.pop()()
        self.assertEqual(feed._on_hp.calls, [])
        self.assertEqual(feed._status_panel_overlay.update.call_count, 0)

    @staticmethod
    def _wait_until(predicate, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()


class _SyncWork:
    """Work queue that executes accepted jobs inline (deterministic tests)."""

    def submit(self, task) -> bool:
        task()
        return True

    def close(self) -> None:
        pass

    @property
    def idle(self) -> bool:
        return True


class _NeverRunsWork:
    """Work queue that accepts jobs but never executes them (stall tests)."""

    def submit(self, task) -> bool:
        return True

    def close(self) -> None:
        pass

    @property
    def idle(self) -> bool:
        return True


class _CountingFeed(PeriodicTaskRunner):
    """Minimal feed that records submit/job/apply calls."""

    def __init__(self, *, submitted: list, applied: list) -> None:
        super().__init__(
            root=object(),
            name="test",
            timeout_s=0.05,
            default_delay_ms=100,
            post_to_tk=lambda cb: cb(),
            log=lambda _msg: None,
        )
        self._submitted = submitted
        self._applied = applied

    def should_submit(self) -> int | None:
        return 100

    def build_job(self, generation: int):
        self._submitted.append(generation)

        def _job() -> None:
            self.publish(generation, "ok")

        return _job

    def apply_result(self, result) -> None:
        self._applied.append(result)


class PeriodicTaskRunnerTests(unittest.TestCase):
    def test_request_submits_and_applies_result(self) -> None:
        submitted: list = []
        applied: list = []
        feed = _CountingFeed(submitted=submitted, applied=applied)
        feed._work = _SyncWork()
        self.assertEqual(feed.request(), 100)
        self.assertEqual(feed._submitted, [1])
        # The inline worker delivered and applied the result already.
        self.assertFalse(feed._pending)
        self.assertEqual(applied, ["ok"])

    def test_reset_drops_stale_result(self) -> None:
        submitted: list = []
        applied: list = []
        feed = _CountingFeed(submitted=submitted, applied=applied)
        feed._work = _NeverRunsWork()
        feed.request()  # in-flight read, generation 1
        feed.reset()    # window changed: generation 2, pending cleared
        # The stale worker finally delivers its generation-1 result.
        feed.publish(1, "ok")
        feed.consume_results()
        self.assertEqual(applied, [])
        self.assertFalse(feed._pending)

    def test_stale_result_cannot_evict_newer_result(self) -> None:
        submitted: list = []
        applied: list = []
        feed = _CountingFeed(submitted=submitted, applied=applied)
        feed._work = _NeverRunsWork()
        feed.request()  # generation 1
        feed.reset()    # invalidate the first native read
        feed.request()  # generation 3, now pending

        # The old native read may finish before the current one. It must not
        # occupy the single result slot or leave generation 3 pending.
        feed.publish(1, "stale")
        feed.publish(3, "current")
        feed.consume_results()

        self.assertEqual(applied, ["current"])
        self.assertFalse(feed._pending)

    def test_pending_request_recovers_after_stall_timeout(self) -> None:
        submitted: list = []
        applied: list = []
        feed = _CountingFeed(submitted=submitted, applied=applied)
        feed._work = _NeverRunsWork()
        feed.request()
        # No job completes; after the stall timeout the worker is abandoned
        # and a fresh one is created so the next tick retries.
        feed._started_at = time.monotonic() - 1.0
        feed.request()
        self.assertFalse(feed._pending)
        self.assertEqual(feed._stall_count, 1)
        self.assertGreaterEqual(feed.request(), 100)
        feed.close()

    def test_result_at_read_deadline_is_consumed_before_stall_recovery(self) -> None:
        submitted: list = []
        applied: list = []
        feed = _CountingFeed(submitted=submitted, applied=applied)
        feed._work = _NeverRunsWork()
        feed.request()  # generation 1
        # Model the bounded read publishing its timeout result at the same
        # instant the watchdog would otherwise inspect the pending request.
        feed._started_at = time.monotonic() - feed._timeout_s
        feed.publish(1, "read_timeout")
        feed.request()
        self.assertEqual(applied, ["read_timeout"])
        self.assertEqual(feed._stall_count, 0)
        # Consuming the timeout immediately starts the next periodic read;
        # that fresh request is expected to be pending here.
        self.assertTrue(feed._pending)
        self.assertEqual(feed._submitted, [1, 2])
        feed.close()

    def test_result_from_abandoned_stall_is_ignored(self) -> None:
        submitted: list = []
        applied: list = []
        feed = _CountingFeed(submitted=submitted, applied=applied)
        feed._work = _NeverRunsWork()
        feed.request()  # generation 1, then abandon it
        feed._started_at = time.monotonic() - 1.0
        feed.request()
        feed.publish(1, "late")
        feed.consume_results()
        self.assertEqual(applied, [])


if __name__ == "__main__":
    unittest.main()
