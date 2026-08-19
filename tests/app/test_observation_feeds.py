"""Unit tests for the extracted observation feeds (no Tk root required)."""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace

from pybot.game_state import PlayerVitals
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
        feed.apply_result((123, snap, None))
        self.assertEqual(feed._on_name.calls, [("Hero",)])
        self.assertEqual(feed._on_sp.calls, [("80/100",)])
        self.assertEqual(feed._on_weight.calls, [("40/50",)])
        self.assertEqual(feed._vitals.sp, (80, 100))
        self.assertEqual(feed._vitals.weight, (40, 50))

    def test_inflight_memory_result_from_before_teleport_cannot_publish(self) -> None:
        """A completed pre-TP memory read cannot restore stale SP."""
        vitals = PlayerVitals()
        feed = _memory_feed()
        feed._vitals = vitals
        old_epoch = vitals.observation_epoch
        vitals.begin_observation_epoch()
        snap = SimpleNamespace(
            ok=True,
            char_name="Hero",
            sp=574,
            sp_max=1454,
            weight=40,
            weight_max=50,
        )
        feed.apply_result((123, snap, old_epoch))
        self.assertIsNone(vitals.sp)
        self.assertEqual(feed._on_sp.calls, [])

    def test_failed_memory_read_from_old_epoch_does_not_clear_fresh_sp(self) -> None:
        """A late failed poll from the previous area must not wipe landing SP."""
        vitals = PlayerVitals()
        feed = _memory_feed()
        feed._vitals = vitals
        old_epoch = vitals.observation_epoch
        epoch = vitals.begin_observation_epoch()
        self.assertTrue(vitals.complete_observation_epoch(epoch))
        self.assertTrue(vitals.publish_sp_if_current(350, 1454, epoch))
        feed.apply_result((123, SimpleNamespace(ok=False), old_epoch))
        self.assertEqual(vitals.sp_pair(), (350, 1454))
        self.assertEqual(feed._on_sp.calls, [])

    def test_ignores_result_for_another_window(self) -> None:
        feed = _memory_feed()
        snap = SimpleNamespace(ok=True, char_name="Hero", sp=1, sp_max=2,
                               weight=3, weight_max=4)
        feed.apply_result((999, snap, None))
        self.assertEqual(feed._on_name.calls, [])
        self.assertEqual(feed._vitals.sp, None)


def _status_feed(**overrides) -> StatusPanelFeed:
    labels = {name: _Recorder() for name in ("hp", "sp", "weight")}
    feed = StatusPanelFeed(
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

    def test_read_failed_permanently_faults_reader(self) -> None:
        """A producer failure cannot be silently restarted by a later start."""
        feed = _status_feed()
        reads = 0
        read_started = threading.Event()

        def read_snapshot():
            nonlocal reads
            reads += 1
            read_started.set()
            return SimpleNamespace(
                hwnd=123,
                state="read_failed",
                error="native parser failed",
            )

        feed._read_snapshot = read_snapshot
        try:
            feed.start()
            self.assertTrue(read_started.wait(timeout=1.0))
            self.assertTrue(self._wait_until(lambda: feed.idle))
            self.assertTrue(feed.faulted)
            self.assertTrue(feed._stopped)
            feed.start()
            self.assertEqual(reads, 1)
        finally:
            feed.close()

    def test_failed_frames_retain_last_published_values(self) -> None:
        """Misses do not clear storage or trigger fallback/re-anchor state."""
        feed = _status_feed()
        values = SimpleNamespace(
            panel_origin=(1, 2),
            hp=90, hp_max=100, sp=70, sp_max=100,
            weight=30, weight_max=50,
        )
        good = SimpleNamespace(hwnd=123, state="values", values=values)
        feed._record_reader_result(good, feed._lifecycle_epoch)
        for state in ("panel_missing", "read_timeout", "roi_missing"):
            feed._record_reader_result(
                SimpleNamespace(hwnd=123, state=state), feed._lifecycle_epoch
            )
        self.assertEqual(feed._vitals.sp, (70, 100))
        self.assertEqual(feed._vitals.clear_count, 0)

    def test_live_read_does_not_enter_bounded_helper_thread(self) -> None:
        """The permanent reader must call the parser directly.

        The feed already owns one long-lived reader thread. Routing every poll
        through the compatibility bounded helper added a second daemon thread
        and a process-wide single-flight gate; after teleport that gate could
        keep returning read_timeout instead of reaching the fixed ROI parser.
        """
        feed = _status_feed()
        direct = SimpleNamespace(hwnd=123, state="values", values=None)
        with patch(
            "pybot.app.status_panel_feed.read_status_panel_snapshot",
            return_value=direct,
        ) as reader:
            result = feed._read_snapshot()
        reader.assert_called_once_with(
            123,
            None,
            refresh_max=True,
            timeout_s=6.0,
            client_hint=None,
            refresh_client=False,
            allow_partial=False,
        )
        self.assertIs(result, direct)
        # The producer has no consumer-driven search flags to consume.
        self.assertIsNone(getattr(feed, "_status_panel_reanchor", None))
        self.assertIsNone(getattr(feed, "_status_panel_geometry_refresh", None))

    def test_inflight_reader_result_from_before_teleport_cannot_publish(self) -> None:
        """A completed pre-TP OCR frame cannot restore stale SP after reset."""
        vitals = PlayerVitals()
        feed = _status_feed()
        feed._vitals = vitals
        values = SimpleNamespace(
            panel_origin=(4, 5),
            hp=90, hp_max=100, sp=574, sp_max=1454,
            weight=20, weight_max=100,
        )
        result = SimpleNamespace(
            hwnd=123, state="values", values=values,
            client_left=10, client_top=20,
            client_width=300, client_height=200,
            full_refresh=True,
        )
        old_epoch = vitals.observation_epoch
        vitals.begin_observation_epoch()
        feed._record_reader_result(
            result,
            feed._lifecycle_epoch,
            observation_epoch=old_epoch,
        )
        self.assertIsNone(vitals.sp)

    def test_teleport_epoch_keeps_static_ocr_anchor(self) -> None:
        """Danger state changes do not force the static panel to re-anchor."""
        vitals = PlayerVitals()
        feed = _status_feed()
        feed._vitals = vitals
        anchor = SimpleNamespace(
            panel_origin=(4, 5),
            hp=90, hp_max=100, sp=574, sp_max=1454,
            weight=20, weight_max=100,
        )
        feed._status_panel_confirmed = anchor
        feed._status_panel_client_hint = (10, 20, 300, 200)
        feed._status_panel_max_read_at = 123.0

        epoch = vitals.begin_observation_epoch()
        feed._sync_observation_epoch(epoch)

        self.assertIs(feed._status_panel_confirmed, anchor)
        self.assertEqual(feed._status_panel_client_hint, (10, 20, 300, 200))
        self.assertEqual(feed._status_panel_max_read_at, 0.0)

        # A stale transition frame is still rejected by the vitals epoch gate
        # and cannot replace the retained static anchor.
        transition_values = SimpleNamespace(
            panel_origin=(99, 101),
            hp=1, hp_max=100, sp=999, sp_max=1454,
            weight=99, weight_max=100,
        )
        feed._record_reader_result(
            SimpleNamespace(
                hwnd=123, state="values", values=transition_values,
                client_left=99, client_top=101,
                client_width=640, client_height=480,
                full_refresh=True,
            ),
            feed._lifecycle_epoch,
            observation_epoch=epoch,
        )
        self.assertIsNone(vitals.sp)
        self.assertIs(feed._status_panel_confirmed, anchor)
        self.assertEqual(feed._status_panel_client_hint, (10, 20, 300, 200))

        # The next read still receives the retained fixed-ROI anchor rather
        # than being forced into a full-client header search.
        with patch(
            "pybot.app.status_panel_feed.read_status_panel_snapshot",
            return_value=SimpleNamespace(hwnd=123, state="roi_missing"),
        ) as reader:
            feed._read_snapshot()
        self.assertIs(reader.call_args.args[1], anchor)
        self.assertEqual(reader.call_args.kwargs["client_hint"], (10, 20, 300, 200))

        self.assertTrue(vitals.complete_observation_epoch(epoch))
        fresh_values = SimpleNamespace(
            panel_origin=(4, 5),
            hp=90, hp_max=100, sp=350, sp_max=1454,
            weight=20, weight_max=100,
        )
        feed._record_reader_result(
            SimpleNamespace(
                hwnd=123, state="values", values=fresh_values,
                client_left=10, client_top=20,
                client_width=300, client_height=200,
                full_refresh=False,
            ),
            feed._lifecycle_epoch,
            observation_epoch=epoch,
        )
        self.assertEqual(vitals.sp_pair(), (350, 1454))
        self.assertIs(feed._status_panel_confirmed, fresh_values)

    def test_live_reader_publishes_fresh_sp_after_teleport_misses(self) -> None:
        """SP publication resumes after the sit/danger-TP/sit gap.

        The game panel is static; transient teleport frames may fail, but they
        must not terminate the producer or make the next fresh SP invisible to
        PlayerVitals.
        """
        feed = _status_feed()
        values = SimpleNamespace(
            panel_origin=(4, 5),
            hp=90, hp_max=100, sp=12, sp_max=100,
            weight=20, weight_max=100,
        )
        results = iter(
            [
                SimpleNamespace(
                    hwnd=123, state="values", values=values,
                    client_left=10, client_top=20,
                    client_width=300, client_height=200,
                    full_refresh=True,
                ),
                SimpleNamespace(hwnd=123, state="panel_missing"),
                SimpleNamespace(hwnd=123, state="read_timeout"),
                SimpleNamespace(hwnd=123, state="sp_only", values=None, sp=(42, 100)),
                SimpleNamespace(
                    hwnd=123, state="values", values=SimpleNamespace(
                        panel_origin=(4, 5),
                        hp=90, hp_max=100, sp=77, sp_max=100,
                        weight=20, weight_max=100,
                    ),
                    client_left=10, client_top=20,
                    client_width=300, client_height=200,
                    full_refresh=False,
                ),
            ]
        )
        with patch.object(feed, "_read_snapshot", side_effect=results):
            for _ in range(5):
                result = feed._read_snapshot()
                feed._record_reader_result(result, feed._lifecycle_epoch)
        self.assertEqual(feed._vitals.sp, (77, 100))
        self.assertEqual(feed._vitals.clear_count, 0)

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

    def discard_pending(self) -> None:
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

    def discard_pending(self) -> None:
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
        feed._work.close()
        feed._work = _SyncWork()
        self.assertEqual(feed.request(), 100)
        self.assertEqual(feed._submitted, [1])
        # The inline worker delivered and applied the result already.
        self.assertFalse(feed._pending)
        self.assertEqual(applied, ["ok"])

    def test_stop_drops_late_result(self) -> None:
        submitted: list = []
        applied: list = []
        feed = _CountingFeed(submitted=submitted, applied=applied)
        feed._work.close()
        feed._work = _NeverRunsWork()
        feed.request()
        feed.stop()
        feed.publish(1, "late")
        feed.consume_results()
        self.assertEqual(applied, [])
        self.assertTrue(feed._stopped)

    def test_reset_drops_stale_result(self) -> None:
        submitted: list = []
        applied: list = []
        feed = _CountingFeed(submitted=submitted, applied=applied)
        feed._work.close()
        feed._work = _NeverRunsWork()
        feed.request()  # in-flight read, generation 1
        feed.reset()    # window changed: generation 2, pending cleared
        # The stale worker finally delivers its generation-1 result.
        feed.publish(1, "ok")
        feed.consume_results()
        self.assertEqual(applied, [])
        # Reset invalidates the old result but keeps the runner pending until
        # the original worker actually exits.
        self.assertTrue(feed._pending)
        feed._worker_finished(1)
        self.assertFalse(feed._pending)

    def test_stale_result_cannot_evict_newer_result(self) -> None:
        submitted: list = []
        applied: list = []
        feed = _CountingFeed(submitted=submitted, applied=applied)
        feed._work.close()
        feed._work = _NeverRunsWork()
        feed.request()  # generation 1
        feed.reset()    # invalidate the first native read
        feed._worker_finished(1)
        feed.request()  # generation 3, now pending

        # The old native read may finish before the current one. It must not
        # occupy the single result slot or leave generation 3 pending.
        feed.publish(1, "stale")
        feed.publish(3, "current")
        feed.consume_results()

        self.assertEqual(applied, ["current"])
        self.assertFalse(feed._pending)

    def test_pending_request_becomes_terminal_fault_at_timeout(self) -> None:
        submitted: list = []
        applied: list = []
        feed = _CountingFeed(submitted=submitted, applied=applied)
        feed._work.close()
        feed._work = _NeverRunsWork()
        feed.request()
        # No job completes; after the timeout the feed must fail closed
        # instead of queuing a retry behind the blocked operation.
        feed._started_at = time.monotonic() - 1.0
        feed.request()
        self.assertFalse(feed._pending)
        self.assertTrue(feed.faulted)
        submitted_count = len(submitted)
        self.assertGreaterEqual(feed.request(), 100)
        self.assertEqual(len(submitted), submitted_count)
        feed.close()

    def test_result_at_read_deadline_is_consumed_before_terminal_fault(self) -> None:
        submitted: list = []
        applied: list = []
        feed = _CountingFeed(submitted=submitted, applied=applied)
        feed._work.close()
        feed._work = _NeverRunsWork()
        feed.request()  # generation 1
        # Model the bounded read publishing its timeout result at the same
        # instant the watchdog would otherwise inspect the pending request.
        feed._started_at = time.monotonic() - feed._timeout_s
        feed.publish(1, "read_timeout")
        feed.request()
        self.assertEqual(applied, ["read_timeout"])
        self.assertFalse(feed.faulted)
        # Consuming a completed result clears the old request and immediately
        # permits exactly one subsequent read on the normal cadence.
        self.assertTrue(feed._pending)
        self.assertEqual(feed._submitted, [1, 2])
        feed.close()

    def test_result_from_abandoned_stall_is_ignored(self) -> None:
        submitted: list = []
        applied: list = []
        feed = _CountingFeed(submitted=submitted, applied=applied)
        feed._work.close()
        feed._work = _NeverRunsWork()
        feed.request()  # generation 1, then abandon it
        feed._started_at = time.monotonic() - 1.0
        feed.request()
        feed.publish(1, "late")
        feed.consume_results()
        self.assertEqual(applied, [])

    def test_terminal_fault_does_not_strand_blocked_worker(self) -> None:
        """A timed-out worker exits cleanly once its native call returns."""
        started = threading.Event()
        release = threading.Event()
        completed = threading.Event()

        class _BlockingFeed(PeriodicTaskRunner):
            def __init__(self) -> None:
                super().__init__(
                    root=object(),
                    name="test-terminal-worker",
                    timeout_s=0.01,
                    default_delay_ms=100,
                    post_to_tk=lambda cb: cb(),
                    log=lambda _msg: None,
                )

            def should_submit(self) -> int | None:
                return 100

            def build_job(self, generation: int):
                def _job() -> None:
                    started.set()
                    release.wait()
                    completed.set()

                return _job

            def apply_result(self, result) -> None:
                raise AssertionError("blocked test must not publish a result")

        feed = _BlockingFeed()
        work = feed._work
        self.assertIsNotNone(work)
        assert work is not None
        try:
            feed.request()
            self.assertTrue(started.wait(timeout=1.0))
            feed._started_at = time.monotonic() - 1.0
            feed.request()
            self.assertTrue(feed.faulted)
            self.assertTrue(work._thread.is_alive())
        finally:
            release.set()
            feed.close()
            work._thread.join(timeout=2.0)
        self.assertTrue(completed.is_set())
        self.assertFalse(work._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
