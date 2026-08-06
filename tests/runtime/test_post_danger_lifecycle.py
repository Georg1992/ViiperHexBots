"""Post-danger teleport lifecycle boundary regressions."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
from pybot.runtime.capture.window_roi import HuntRoi
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

    def test_sprite_grf_detection_blocks_clear_before_track_creation(self) -> None:
        """GRF detection must block teleport while its candidate awaits tracking."""
        self.ctx.config = _config(use_sprite_grf=True)
        self.ctx.capture.is_valid.return_value = True
        self.ctx.capture.get_hunt_roi.return_value = HuntRoi(
            x=0, y=0, w=200, h=200,
        )
        self.ctx.capture.capture_roi.return_value = SimpleNamespace(size=1)
        self.ctx.detector.discover_frame.return_value = SimpleNamespace(
            ok=True,
            raw_count=1,
            duration_ms=1,
            detections=[
                SimpleNamespace(
                    x=100,
                    y=100,
                    confidence=0.9,
                    candidate_scale=1.0,
                    living=True,
                    bbox=(90, 90, 20, 20),
                )
            ],
            timing={},
            lock_wait_ms=0,
            detect_ms=1,
        )
        summary = SimpleNamespace(
            added_count=0,
            alive_after=0,
            removed_count=0,
            matched_count=0,
            death_sites_active=0,
            created_ids=[],
            removed_ids=[],
            removed_out_of_range_ids=[],
            removed_discovery_miss_ids=[],
            tracks_before=0,
            tracks_after=0,
        )
        with patch.object(
            self.tracks,
            "process_discovery_scan",
            return_value=summary,
        ):
            DiscoveryWorker(self.ctx, self.mode)._scan()

        # Isolate the race: the detector saw a mob, but no track/candidate was
        # available by the time the reconciliation result was published.
        self.assertFalse(self.tracks.has_pending_discovery_candidates())
        self.assertFalse(self.mode.discovery_confirmed_clear)
        self.assertFalse(self.mode.on_no_attackable_targets())

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
