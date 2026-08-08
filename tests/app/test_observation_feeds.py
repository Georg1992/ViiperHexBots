"""Unit tests for the extracted observation feeds (no Tk root required)."""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pybot.app.memory_stats_feed import MEMORY_POLL_MS, MemoryStatsFeed
from pybot.app.periodic_task_runner import PeriodicTaskRunner
from pybot.app.status_panel_feed import (
    STATUS_PANEL_READ_TIMEOUT_S,
    STATUS_PANEL_SEARCH_MS,
    STATUS_PANEL_VALUE_MS,
    StatusPanelFeed,
)


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
    def test_watchdog_waits_after_bounded_read_timeout(self) -> None:
        feed = _status_feed()
        # The watchdog must not share the exact deadline with the bounded
        # native read: the read needs time to publish its timeout result.
        self.assertGreater(feed._timeout_s, STATUS_PANEL_READ_TIMEOUT_S)

    def test_searches_slowly_when_panel_not_confirmed(self) -> None:
        feed = _status_feed()
        self.assertEqual(feed.should_submit(), STATUS_PANEL_SEARCH_MS)

    def test_polls_fast_when_panel_confirmed(self) -> None:
        feed = _status_feed()
        feed._status_panel_confirmed = SimpleNamespace()
        self.assertEqual(feed.should_submit(), STATUS_PANEL_VALUE_MS)

    def test_reads_stop_when_feed_is_inactive(self) -> None:
        feed = _status_feed()
        feed.set_active(False)
        self.assertIsNone(feed.should_submit())
        feed._status_panel_confirmed = SimpleNamespace()
        self.assertIsNone(feed.should_submit())

    def test_reads_resume_when_feed_is_reactivated(self) -> None:
        feed = _status_feed()
        feed.set_active(False)
        feed.set_active(True)
        self.assertEqual(feed.should_submit(), STATUS_PANEL_SEARCH_MS)
        feed._status_panel_confirmed = SimpleNamespace()
        self.assertEqual(feed.should_submit(), STATUS_PANEL_VALUE_MS)

    def test_inactive_feed_drops_completed_result(self) -> None:
        feed = _status_feed()
        feed.set_active(False)
        result = SimpleNamespace(hwnd=123, state="hp_only", hp=(50, 100))
        feed.apply_result(result)
        self.assertEqual(feed._vitals.hp, None)

    def test_hp_only_result_publishes_hp(self) -> None:
        feed = _status_feed()
        result = SimpleNamespace(
            hwnd=123, state="hp_only", hp=(50, 100),
        )
        feed.apply_result(result)
        self.assertEqual(feed._vitals.hp, (50, 100))
        self.assertEqual(feed._on_hp.calls, [("50/100",)])

    def test_sp_only_result_publishes_sp_during_sit_ocr_gap(self) -> None:
        """SP recovery must not depend on HP/Weight succeeding in the frame."""
        feed = _status_feed()
        result = SimpleNamespace(
            hwnd=123, state="sp_only", sp=(98, 100),
        )
        feed.apply_result(result)
        self.assertEqual(feed._vitals.sp, (98, 100))
        self.assertEqual(feed._on_sp.calls, [("98/100",)])

    def test_reanchors_only_after_three_fixed_roi_misses(self) -> None:
        feed = _status_feed()
        for _ in range(2):
            feed.apply_result(SimpleNamespace(
                hwnd=123,
                state="roi_missing",
                client_left=10,
                client_top=20,
                client_width=300,
                client_height=200,
            ))
        self.assertFalse(feed._status_panel_geometry_refresh)
        self.assertFalse(feed._status_panel_reanchor)

        feed.apply_result(SimpleNamespace(
            hwnd=123,
            state="roi_missing",
            client_left=10,
            client_top=20,
            client_width=300,
            client_height=200,
        ))
        self.assertTrue(feed._status_panel_geometry_refresh)
        self.assertTrue(feed._status_panel_reanchor)

    def test_valid_fixed_roi_result_resets_miss_streak(self) -> None:
        feed = _status_feed()
        feed._status_panel_miss_count = 2
        feed.apply_result(SimpleNamespace(
            hwnd=123,
            state="sp_only",
            sp=(98, 100),
        ))
        self.assertEqual(feed._status_panel_miss_count, 0)

    def test_hp_sp_only_result_publishes_both_independently(self) -> None:
        feed = _status_feed()
        result = SimpleNamespace(
            hwnd=123, state="hp_sp_only", hp=(80, 100), sp=(98, 100),
        )
        feed.apply_result(result)
        self.assertEqual(feed._vitals.hp, (80, 100))
        self.assertEqual(feed._vitals.sp, (98, 100))

    def test_panel_missing_clears_vision_stats(self) -> None:
        feed = _status_feed()
        result = SimpleNamespace(
            hwnd=123, state="panel_missing", client_left=10, client_top=20,
        )
        feed.apply_result(result)
        feed._status_panel_overlay.show_panel_missing.assert_called_once()
        self.assertEqual(feed._on_hp.calls, [("—",)])

    def test_committed_values_publish_hp_and_vision_sp_weight(self) -> None:
        feed = _status_feed()
        values = SimpleNamespace(
            panel_origin=(1, 2),
            hp=90, hp_max=100, sp=70, sp_max=100,
            weight=30, weight_max=50,
        )
        result = SimpleNamespace(
            hwnd=123, state="ok", values=values,
            client_left=10, client_top=20, full_refresh=True,
        )
        feed.apply_result(result)
        feed._status_panel_overlay.update.assert_called_once_with(
            values, client_left=10, client_top=20
        )
        self.assertEqual(feed._vitals.hp, (90, 100))
        self.assertEqual(feed._vitals.sp, (70, 100))
        self.assertEqual(feed._on_hp.calls, [("90/100",)])
        self.assertEqual(feed._on_sp.calls, [("70/100",)])

    def test_timeout_then_next_successful_read_restores_status_overlay(self) -> None:
        feed = _status_feed()
        feed.apply_result(SimpleNamespace(
            hwnd=123,
            state="read_timeout",
            error="native status read exceeded timeout",
        ))
        values = SimpleNamespace(
            panel_origin=(1, 2),
            hp=90,
            hp_max=100,
            sp=80,
            sp_max=100,
            weight=30,
            weight_max=50,
        )
        feed.apply_result(SimpleNamespace(
            hwnd=123,
            state="values",
            values=values,
            client_left=10,
            client_top=20,
            full_refresh=True,
        ))
        feed._status_panel_overlay.hide.assert_not_called()
        feed._status_panel_overlay.update.assert_called_once_with(
            values,
            client_left=10,
            client_top=20,
        )
        self.assertEqual(feed._vitals.sp, (80, 100))

    def test_timeout_result_then_fresh_read_through_runner(self) -> None:
        """A timed-out OCR read must not prevent the next read from applying."""
        feed = _status_feed()
        feed._work = _SyncWork()
        values = SimpleNamespace(
            panel_origin=(1, 2),
            hp=90,
            hp_max=100,
            sp=80,
            sp_max=100,
            weight=30,
            weight_max=50,
        )
        with patch(
            "pybot.app.status_panel_feed.read_status_panel_snapshot_bounded",
            side_effect=[
                SimpleNamespace(
                    hwnd=123,
                    state="read_timeout",
                    error="native status read exceeded timeout",
                ),
                SimpleNamespace(
                    hwnd=123,
                    state="values",
                    values=values,
                    client_left=10,
                    client_top=20,
                    full_refresh=True,
                ),
            ],
        ):
            feed.request()
            feed.request()
            feed.request()

        feed._status_panel_overlay.hide.assert_not_called()
        feed._status_panel_overlay.update.assert_called_once_with(
            values,
            client_left=10,
            client_top=20,
        )
        self.assertEqual(feed._vitals.sp, (80, 100))
        self.assertEqual(feed._stall_count, 0)

    def test_timeout_then_hung_retry_then_successful_read_recovers(self) -> None:
        """A second hung read must not block a later fresh OCR result."""
        feed = _status_feed()
        submitted: list[int] = []
        second_started = threading.Event()
        release_second = threading.Event()
        call_count = 0
        first_finished = threading.Event()
        third_finished = threading.Event()

        class _ScriptedWork:
            def submit(self, task) -> bool:
                submitted.append(len(submitted) + 1)
                threading.Thread(
                    target=task,
                    name="test-status-read",
                    daemon=True,
                ).start()
                return True

            def close(self) -> None:
                pass

            @property
            def idle(self) -> bool:
                return True

        values = SimpleNamespace(
            panel_origin=(1, 2),
            hp=90,
            hp_max=100,
            sp=80,
            sp_max=100,
            weight=30,
            weight_max=50,
        )

        def scripted_read(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                first_finished.set()
                return SimpleNamespace(hwnd=123, state="read_timeout")
            if call_count == 2:
                second_started.set()
                release_second.wait(timeout=2.0)
                return SimpleNamespace(hwnd=123, state="read_timeout")
            third_finished.set()
            return SimpleNamespace(
                hwnd=123,
                state="values",
                values=values,
                client_left=10,
                client_top=20,
                full_refresh=True,
            )

        with patch(
            "pybot.app.status_panel_feed.read_status_panel_snapshot_bounded",
            side_effect=scripted_read,
        ):
            feed._work = _ScriptedWork()
            feed.request()  # generation 1 returns read_timeout
            self.assertTrue(first_finished.wait(timeout=1.0))
            feed.request()  # consume timeout; submit the hung generation 2
            self.assertTrue(second_started.wait(timeout=1.0))
            feed._started_at = time.monotonic() - feed._timeout_s - 0.01
            feed.request()  # watchdog abandons generation 2
            self.assertEqual(feed._stall_count, 1)
            # The production runner replaced its queue; keep the scripted
            # queue for this deterministic test so generation 3 can run.
            feed._work = _ScriptedWork()
            feed.request()  # next Tk tick submits generation 3
            self.assertTrue(third_finished.wait(timeout=1.0))
            # The worker posts its result through the test feed's no-op Tk
            # callback; consume it explicitly on this simulated UI tick.
            feed.consume_results()

        release_second.set()
        feed._work.close()
        feed._status_panel_overlay.hide.assert_not_called()
        feed._status_panel_overlay.update.assert_called_once_with(
            values,
            client_left=10,
            client_top=20,
        )
        self.assertEqual(submitted, [1, 2, 3])

    def test_reset_forces_fresh_header_search(self) -> None:
        """A window/client change must forget the confirmed panel layout."""
        feed = _status_feed()
        feed._status_panel_confirmed = SimpleNamespace()
        feed._status_panel_max_read_at = 5.0
        feed.reset()
        self.assertIsNone(feed._status_panel_confirmed)
        self.assertEqual(feed._status_panel_max_read_at, 0.0)
        # Without a confirmed panel the next read searches slowly for the
        # header instead of fast-polling stale coordinates.
        self.assertEqual(feed.should_submit(), STATUS_PANEL_SEARCH_MS)

    def test_memory_mode_does_not_publish_vision_sp_weight(self) -> None:
        feed = _status_feed(use_memory_reading=True)
        values = SimpleNamespace(
            panel_origin=(1, 2),
            hp=90, hp_max=100, sp=70, sp_max=100,
            weight=30, weight_max=50,
        )
        result = SimpleNamespace(
            hwnd=123, state="ok", values=values,
            client_left=0, client_top=0, full_refresh=False,
        )
        feed.apply_result(result)
        self.assertEqual(feed._vitals.sp, None)
        self.assertEqual(feed._vitals.weight, None)
        self.assertEqual(feed._on_sp.calls, [])

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
