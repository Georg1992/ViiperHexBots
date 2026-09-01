"""Unit tests for the extracted observation feeds (no Tk root required)."""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace

from pybot.game_state import PlayerVitals
from unittest.mock import MagicMock, patch

from pybot.app.memory_stats_feed import MemoryReadResult, MemoryStatsFeed
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


def _ok_snap(**overrides) -> SimpleNamespace:
    values = dict(
        ok=True,
        char_name="Hero",
        sp=80,
        sp_max=100,
        weight=40,
        weight_max=50,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _memory_feed(**overrides) -> MemoryStatsFeed:
    labels = {name: _Recorder() for name in ("name", "sp", "weight")}
    feed = MemoryStatsFeed(
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
    def test_inactive_when_memory_reading_disabled(self) -> None:
        feed = _memory_feed(use_memory_reading=False)
        with patch("pybot.app.memory_stats_feed.load_client_profile") as load:
            result = feed._read_snapshot()
        self.assertEqual(result.state, "inactive")
        load.assert_not_called()
        feed._record_reader_result(result, feed._lifecycle_epoch)
        self.assertIsNone(feed._vitals.sp)
        self.assertEqual(feed._vitals.clear_count, 0)

    @patch("pybot.app.memory_stats_feed.window_exists", return_value=True)
    @patch(
        "pybot.app.memory_stats_feed.load_client_profile",
        return_value=SimpleNamespace(memory=SimpleNamespace(has_any=True)),
    )
    def test_reads_when_profile_and_window_ok(self, _load, _window) -> None:
        feed = _memory_feed()
        feed._poller.read.return_value = _ok_snap()
        result = feed._read_snapshot()
        self.assertEqual(result.state, "values")
        self.assertEqual(result.hwnd, 123)
        feed._poller.read.assert_called_once()

    @patch(
        "pybot.app.memory_stats_feed.load_client_profile",
        return_value=SimpleNamespace(memory=SimpleNamespace(has_any=False)),
    )
    def test_inactive_without_profile_addresses_does_not_clear_sp(self, _load) -> None:
        feed = _memory_feed()
        feed._vitals.publish_sp(70, 100)
        result = feed._read_snapshot()
        self.assertEqual(result.state, "inactive")
        feed._record_reader_result(result, feed._lifecycle_epoch)
        self.assertEqual(feed._vitals.sp, (70, 100))
        self.assertEqual(feed._vitals.clear_count, 0)
        feed._project_result(result)
        self.assertEqual(feed._on_name.calls, [("—",)])
        self.assertEqual(feed._on_sp.calls, [("—",)])
        self.assertEqual(feed._on_weight.calls, [("—",)])

    def test_applies_snapshot_to_labels_and_vitals(self) -> None:
        feed = _memory_feed()
        result = MemoryReadResult(hwnd=123, state="values", snap=_ok_snap())
        feed._record_reader_result(result, feed._lifecycle_epoch)
        feed._project_result(result)
        self.assertEqual(feed._on_name.calls, [("Hero",)])
        self.assertEqual(feed._on_sp.calls, [("80/100",)])
        self.assertEqual(feed._on_weight.calls, [("40/50",)])
        self.assertEqual(feed._vitals.sp, (80, 100))
        self.assertEqual(feed._vitals.weight, (40, 50))

    def test_autonomous_reader_publishes_without_tk_callback(self) -> None:
        """Memory/vitals continue even when Tk has not drained presentation."""
        feed = _memory_feed()
        vitals = PlayerVitals()
        feed._vitals = vitals
        read_started = threading.Event()
        callbacks: list = []
        snap = _ok_snap(sp=70, sp_max=100, weight=30, weight_max=50)

        def read_snapshot():
            read_started.set()
            return MemoryReadResult(hwnd=123, state="values", snap=snap)

        feed._read_snapshot = read_snapshot
        feed._post_to_tk = callbacks.append
        try:
            feed.start()
            self.assertTrue(read_started.wait(timeout=1.0))
            self.assertTrue(self._wait_until(lambda: vitals.sp == 70))
            self.assertEqual(vitals.sp_pair(), (70, 100))
            self.assertEqual(feed._on_sp.calls, [])
            self.assertEqual(len(callbacks), 1)
        finally:
            feed.close()
            self.assertTrue(self._wait_until(lambda: feed.idle))

    def test_reader_exception_permanently_faults_reader(self) -> None:
        feed = _memory_feed()
        reads = 0
        read_started = threading.Event()

        def read_snapshot():
            nonlocal reads
            reads += 1
            read_started.set()
            raise RuntimeError("native read exploded")

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
        """Misses do not clear storage or switch to another source."""
        feed = _memory_feed()
        good = MemoryReadResult(
            hwnd=123, state="values", snap=_ok_snap(sp=70, sp_max=100)
        )
        feed._record_reader_result(good, feed._lifecycle_epoch)
        for state in ("inactive", "failed"):
            feed._record_reader_result(
                MemoryReadResult(
                    hwnd=123, state=state, snap=SimpleNamespace(ok=False)
                ),
                feed._lifecycle_epoch,
            )
        self.assertEqual(feed._vitals.sp, (70, 100))
        self.assertEqual(feed._vitals.clear_count, 0)

    def test_inflight_memory_result_from_before_teleport_cannot_publish(self) -> None:
        """A completed pre-TP memory read cannot restore stale SP."""
        vitals = PlayerVitals()
        feed = _memory_feed()
        feed._vitals = vitals
        old_epoch = vitals.observation_epoch
        vitals.begin_observation_epoch()
        feed._record_reader_result(
            MemoryReadResult(
                hwnd=123, state="values", snap=_ok_snap(sp=574, sp_max=1454)
            ),
            feed._lifecycle_epoch,
            observation_epoch=old_epoch,
        )
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
        feed._record_reader_result(
            MemoryReadResult(
                hwnd=123, state="failed", snap=SimpleNamespace(ok=False)
            ),
            feed._lifecycle_epoch,
            observation_epoch=old_epoch,
        )
        self.assertEqual(vitals.sp_pair(), (350, 1454))
        self.assertEqual(feed._on_sp.calls, [])

    def test_ignores_result_for_another_window(self) -> None:
        feed = _memory_feed()
        feed._record_reader_result(
            MemoryReadResult(
                hwnd=999, state="values", snap=_ok_snap(sp=1, sp_max=2)
            ),
            feed._lifecycle_epoch,
        )
        self.assertEqual(feed._on_name.calls, [])
        self.assertEqual(feed._vitals.sp, None)

    @staticmethod
    def _wait_until(predicate, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()


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



if __name__ == "__main__":
    unittest.main()
