from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from pybot.app.log_pipe import LogPipe
from pybot.runtime.hunt_tracks import HuntTracks
from pybot.runtime.logging import HuntLogger
from pybot.runtime.workers.coord_tracking_worker import CoordTrackingWorker
from pybot.recognition.detector.detector import load_detector_config


class _Root:
    def after(self, *_args) -> None:
        return None


class RuntimeResourceBoundsTests(unittest.TestCase):
    def test_log_pipe_queue_is_bounded_without_blocking(self) -> None:
        pipe = LogPipe(_Root())
        for index in range(5000):
            pipe.log(f"line-{index}")
        self.assertLessEqual(pipe._queue.qsize(), 2000)

    def test_logger_instances_with_same_session_id_are_independent(self) -> None:
        first = HuntLogger(session_id="resource_bounds_test", echo_stdout=False)
        second = HuntLogger(session_id="resource_bounds_test", echo_stdout=False)
        first_name = first._behavior.name
        second_name = second._behavior.name
        try:
            self.assertIsNot(first._behavior, second._behavior)
            self.assertIsNot(first._listener, second._listener)
        finally:
            first.close()
            second.close()
        self.assertFalse(first._listener)
        self.assertFalse(second._listener)
        self.assertNotIn(first_name, first._behavior.manager.loggerDict)
        self.assertNotIn(second_name, second._behavior.manager.loggerDict)

    def test_tracking_diagnostic_key_includes_area_epoch(self) -> None:
        tracks = HuntTracks(load_detector_config())
        ctx = MagicMock()
        ctx.tracks = tracks
        ctx.discovery_suspend = threading.Event()
        ctx.discovery_wake = threading.Event()
        ctx.capture.is_valid.return_value = True
        ctx.capture.get_hunt_roi.return_value = SimpleNamespace(
            x=0, y=0, w=200, h=200,
        )
        ctx.capture.capture_roi.return_value = SimpleNamespace(size=1)
        ctx.tracker.track_locals_frame.return_value = SimpleNamespace(results=[])
        ctx.config.mob_name = "horn"
        ctx.config.use_sprite_grf = True
        worker = CoordTrackingWorker(ctx)

        first = tracks.create_track("horn", 100, 100, 0.8, 0.9, now_tick=1)
        worker._logged_first_tick.add((tracks.area_epoch, first.id))
        tracks.area_reset()
        second = tracks.create_track("horn", 100, 100, 0.8, 0.9, now_tick=2)
        self.assertEqual(first.id, second.id)
        worker._update_overlay(1000)
        self.assertNotIn((0, second.id), worker._logged_first_tick)
        self.assertEqual(tracks.area_epoch, 1)


if __name__ == "__main__":
    unittest.main()
