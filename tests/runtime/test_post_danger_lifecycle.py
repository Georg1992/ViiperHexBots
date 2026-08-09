"""Post-danger teleport lifecycle boundary regressions."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from pybot.config.runtime import HuntRuntimeConfig
from pybot.paths import PROJECT_ROOT
from pybot.runtime.control import RuntimeControl
from pybot.runtime.hunt_mode import create_hunt_mode
from pybot.runtime.hunt_policy import HuntPolicy
from pybot.runtime.hunt_tracks import HuntTracks
from pybot.runtime.input.input_backend import ShadowInputBackend
from pybot.runtime.logging import HuntLogger
from pybot.runtime.overlay_ports import NullOverlay
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.validation_log import HuntValidationLogger
from pybot.runtime.teleport import TeleportController
from pybot.runtime.workers.discovery_worker import DiscoveryWorker


def _config(
    *,
    teleport_duration_ms: int = 10,
    use_sprite_grf: bool = False,
) -> HuntRuntimeConfig:
    return HuntRuntimeConfig(
        config_path=PROJECT_ROOT / "config.ini",
        hwnd=1,
        mob_name="horn",
        hunt_mode="teleport",
        skill_delay_ms=500,
        skill_button="e",
        skill_scan_code=18,
        teleport_button="q",
        teleport_scan_code=16,
        search_range_cells=16,
        cell_size_px=64,
        discovery_interval_ms=250,
        teleport_duration_ms=teleport_duration_ms,
        validation_enabled=False,
        control_file=None,
        use_sprite_grf=use_sprite_grf,
    )


class PostDangerLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = HuntLogger(session_id="test_post_danger_lifecycle")
        self.tracks = HuntTracks()
        self.ctx = HuntRuntimeContext(
            config=_config(),
            logger=self.logger,
            tracks=self.tracks,
            policy=HuntPolicy(),
            capture=MagicMock(),
            detector=MagicMock(),
            tracker=MagicMock(),
            validation=HuntValidationLogger(self.logger, self.tracks, enabled=False),
            control=RuntimeControl(None),
            overlay=NullOverlay(),
        )
        self.ctx.mark_running()
        self.mode = create_hunt_mode(self.ctx, ShadowInputBackend())

    def tearDown(self) -> None:
        self.logger.close()

    def test_reset_clears_strategy_before_new_screen_is_observable(self) -> None:
        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=self.tracks.area_epoch,
        )
        self.assertTrue(self.mode.discovery_since_reset)

        with self.ctx.area_transition_lock:
            self.mode.on_area_reset()
            self.ctx.area_reset(reason="danger_teleport")
            self.assertFalse(self.mode.discovery_since_reset)
            self.assertFalse(self.mode.discovery_confirmed_clear)
            self.assertEqual(self.tracks.get_area_clear_candidate().alive_count, 0)

        # No-target handling cannot treat the new area as discovery-confirmed.
        self.assertFalse(self.mode.on_no_attackable_targets())

    def test_teleport_clears_tracks_before_settle(self) -> None:
        """Accepted teleport input removes old targets before landing wait."""
        track = self.tracks.create_track(
            "horn", 100, 100, 0.9, 0.8, now_tick=1,
        )
        self.assertIsNotNone(track)
        settle_checked = threading.Event()

        def wait_for_settle(_timeout_s: float) -> bool:
            # The old area must already be empty while the client is still in
            # its landing transition. This is the important danger/sit race
            # boundary: gameplay cannot keep consuming the old track.
            self.assertEqual(self.tracks.get_track_count(), 0)
            settle_checked.set()
            return True

        self.ctx.wait_unless_stopped = wait_for_settle  # type: ignore[method-assign]
        teleport = TeleportController(self.ctx, ShadowInputBackend(), self.mode)

        self.assertTrue(teleport.teleport_once(scan_code=16))
        self.assertTrue(settle_checked.is_set())
        self.assertEqual(self.tracks.get_track_count(), 0)

    def test_late_tracking_result_from_old_epoch_is_ignored(self) -> None:
        """A callback completing after teleport cannot update the new area."""
        track = self.tracks.create_track(
            "horn", 100, 100, 0.9, 0.8, now_tick=1,
        )
        assert track is not None
        old_epoch = self.tracks.area_epoch
        teleport = TeleportController(self.ctx, ShadowInputBackend(), self.mode)
        self.assertTrue(teleport.teleport_once(scan_code=16))

        result = SimpleNamespace(
            track_id=track.id,
            found=True,
            x=999,
            y=999,
            confidence=1.0,
            opacity_score=0.0,
        )
        missed, deaths = self.tracks.apply_tracking(
            [result],
            now_tick=2,
            area_epoch=old_epoch,
        )
        self.assertEqual(missed, [])
        self.assertEqual(deaths, [])
        self.assertEqual(self.tracks.get_track_count(), 0)

    def test_interrupted_teleport_still_leaves_tracks_cleared(self) -> None:
        """A stopped/interrupted settle cannot resurrect old-area targets."""
        self.tracks.create_track("horn", 100, 100, 0.9, 0.8, now_tick=1)
        self.ctx.wait_unless_stopped = MagicMock(return_value=False)
        teleport = TeleportController(self.ctx, ShadowInputBackend(), self.mode)

        self.assertFalse(teleport.teleport_once(scan_code=16))
        self.assertEqual(self.tracks.get_track_count(), 0)

    def test_inflight_old_discovery_cannot_publish_after_area_reset(self) -> None:
        self.ctx.capture.is_valid.return_value = True
        self.ctx.capture.get_hunt_roi.return_value = SimpleNamespace(
            x=0, y=0, w=200, h=200,
        )
        self.ctx.capture.capture_roi.return_value = SimpleNamespace(size=1)
        self.ctx.detector.discover_frame.side_effect = lambda _frame, _roi: (
            SimpleNamespace(ok=True, raw_count=1, duration_ms=1, detections=[])
        )
        worker = DiscoveryWorker(self.ctx, self.mode)

        entered_detector = threading.Event()
        release_detector = threading.Event()

        def blocked_scan(_frame, _roi):
            entered_detector.set()
            release_detector.wait(timeout=1.0)
            return SimpleNamespace(ok=True, raw_count=1, duration_ms=1, detections=[])

        self.ctx.detector.discover_frame.side_effect = blocked_scan
        thread = threading.Thread(target=worker._scan, daemon=True)
        thread.start()
        self.assertTrue(entered_detector.wait(timeout=1.0))

        tport = TeleportController(self.ctx, ShadowInputBackend(), self.mode)
        self.ctx.wait_unless_stopped = MagicMock(return_value=True)
        # The real danger path must own the transition boundary while the old
        # discovery call is still blocked.
        self.assertTrue(tport.danger_teleport(reason="critical_hunt"))
        release_detector.set()
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())

        self.assertFalse(self.mode.discovery_since_reset)
        self.assertFalse(self.mode.discovery_confirmed_clear)
        self.assertFalse(self.tracks.has_pending_discovery_candidates())


if __name__ == "__main__":
    unittest.main()
