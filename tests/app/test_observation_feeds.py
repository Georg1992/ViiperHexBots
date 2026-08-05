"""Unit tests for the extracted observation feeds (no Tk root required)."""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pybot.app.memory_stats_feed import MEMORY_POLL_MS, MemoryStatsFeed
from pybot.app.periodic_task_runner import PeriodicTaskRunner
from pybot.app.status_panel_feed import (
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


def _status_feed(**overrides) -> StatusPanelFeed:
    labels = {name: _Recorder() for name in ("hp", "sp", "weight")}
    feed = StatusPanelFeed(
        root=object(),
        config=SimpleNamespace(window_id=123, use_memory_reading=False),
        vitals=_FakeVitals(),
        overlay=MagicMock(),
        panel_active=lambda: True,
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
    def test_searches_slowly_when_panel_not_confirmed(self) -> None:
        feed = _status_feed()
        self.assertEqual(feed.should_submit(), STATUS_PANEL_SEARCH_MS)

    def test_polls_fast_when_panel_confirmed(self) -> None:
        feed = _status_feed()
        feed._status_panel_confirmed = SimpleNamespace()
        self.assertEqual(feed.should_submit(), STATUS_PANEL_VALUE_MS)

    def test_skips_when_session_inactive_and_hides_overlay(self) -> None:
        feed = _status_feed()
        feed._panel_active = lambda: False
        self.assertIsNone(feed.should_submit())
        feed._status_panel_overlay.hide.assert_called_once_with()

    def test_hp_only_result_publishes_hp(self) -> None:
        feed = _status_feed()
        result = SimpleNamespace(
            hwnd=123, state="hp_only", hp=(50, 100),
        )
        feed.apply_result(result)
        self.assertEqual(feed._vitals.hp, (50, 100))
        self.assertEqual(feed._on_hp.calls, [("50/100",)])

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
