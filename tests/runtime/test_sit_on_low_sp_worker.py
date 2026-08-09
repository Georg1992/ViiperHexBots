"""Sit/stand: one press each, no pose-driven re-toggle; hunt until SP recovers."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import (
    SIT_KEY_SETTLE_S,
    SIT_LOW_SP_RATIO,
    SIT_POST_TELEPORT_SETTLE_S,
    SIT_RESUME_SP_RATIO,
    SIT_STAND_RESUME_DELAY_S,
)
from pybot.runtime.danger_detector import DangerDetector, DangerLevel
from pybot.runtime.input.input_backend import ShadowInputBackend
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.workers.attack_loop import GameplayLoop
from pybot.runtime.workers.sit_on_low_sp_worker import SitOnLowSpWorker


class _ScriptedVitals(PlayerVitals):
    def __init__(self, ratios: list[float | None]) -> None:
        super().__init__()
        self._ratios = list(ratios)

    def sp_pair(self) -> tuple[int | None, int | None]:
        if not self._ratios:
            return 98, 100
        ratio = self._ratios.pop(0)
        if ratio is None:
            return None, None
        return int(ratio * 100), 100

    @property
    def observed_ms(self) -> int:
        # Scripted samples are always fresh — never mistaken for a stale feed.
        return int(time.monotonic() * 1000)

    @property
    def changed_ms(self) -> int:
        return int(time.monotonic() * 1000)


class SitOnLowSpWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MagicMock()
        self.config.hwnd = 1
        self.config.sit_on_low_sp_button = "insert"
        self.config.sit_on_low_sp_scan_code = 82
        self.config.teleport_button = "q"
        self.config.teleport_scan_code = 16
        self.config.teleport_duration_ms = 10
        self.config.cell_size_px = 64
        self.config.creamy_tp_button = "w"
        self.config.creamy_tp_scan_code = 17
        self.config.take_fly_wings = False
        self.config.open_storage_steps = ()
        self.ctx = HuntRuntimeContext(
            config=self.config,
            logger=MagicMock(),
            tracks=MagicMock(),
            policy=MagicMock(),
            capture=MagicMock(),
            detector=MagicMock(),
            tracker=MagicMock(),
            validation=MagicMock(),
            control=MagicMock(),
            overlay=MagicMock(),
        )
        self.input = MagicMock(spec=ShadowInputBackend)
        from pybot.runtime.teleport import TeleportController
        self.teleport = TeleportController(self.ctx, self.input, MagicMock())
        self.teleport.teleport_until_quiet = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.danger = MagicMock(spec=DangerDetector)
        self.danger.danger_level.return_value = DangerLevel.SAFE
        self.ctx.danger_detector = self.danger

    def _worker(self, vitals: PlayerVitals | None = None) -> SitOnLowSpWorker:
        return SitOnLowSpWorker(
            self.ctx, self.input, self.teleport,
            danger=self.danger, vitals=vitals or _ScriptedVitals([]),
        )

    def test_thresholds(self) -> None:
        self.assertAlmostEqual(SIT_LOW_SP_RATIO, 0.05)
        self.assertAlmostEqual(SIT_STAND_RESUME_DELAY_S, 0.6)
        self.assertAlmostEqual(SIT_RESUME_SP_RATIO, 0.98)
        self.assertGreaterEqual(SIT_KEY_SETTLE_S, 0.3)

    def test_sit_presses_once_and_marks_seated(self) -> None:
        worker = self._worker()
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.assertTrue(worker.sit(82))
        self.input.toggle_key.assert_called_once_with(82)
        self.assertTrue(worker._seated)
        # Second sit is no-op — no flap.
        self.assertTrue(worker.sit(82))
        self.input.toggle_key.assert_called_once_with(82)

    def test_rejected_sit_does_not_mark_seated_or_retry_toggle(self) -> None:
        worker = self._worker()
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.input.toggle_key.return_value = False

        self.assertFalse(worker.sit(82))
        self.assertFalse(worker._seated)
        self.assertFalse(worker.sit(82))
        self.input.toggle_key.assert_has_calls([
            unittest.mock.call(82),
            unittest.mock.call(82),
        ])

    def test_interrupted_settle_after_accepted_sit_keeps_state_seated(self) -> None:
        worker = self._worker()
        self.ctx.wait_unless_stopped = lambda _t: False  # type: ignore[method-assign]
        self.input.toggle_key.return_value = True

        self.assertTrue(worker.sit(82))
        self.assertTrue(worker._seated)
        self.input.toggle_key.assert_called_once_with(82)

    def test_rejected_stand_keeps_state_for_cleanup_retry(self) -> None:
        worker = self._worker()
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._seated = True
        self.input.toggle_key.side_effect = [False, True]

        self.assertFalse(worker.stand(82))
        self.assertTrue(worker._seated)
        self.assertTrue(worker.stand(82))
        self.assertFalse(worker._seated)
        self.assertEqual(
            [call.args[0] for call in self.input.toggle_key.call_args_list],
            [82, 82],
        )

    def test_shutdown_cleanup_stands_after_stop_without_normal_toggle(self) -> None:
        worker = self._worker()
        self.ctx.stop_event.set()
        worker._seated = True
        self.input.cleanup_toggle_key.return_value = True

        self.assertTrue(worker._cleanup_stand(82))
        worker._seated = False
        self.input.cleanup_toggle_key.assert_called_once_with(82)

    def test_stand_presses_once_and_clears_seated(self) -> None:
        worker = self._worker()
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._seated = True
        self.assertTrue(worker.stand(82))
        self.input.toggle_key.assert_called_once_with(82)
        self.assertFalse(worker._seated)
        # Second stand is no-op — no flap.
        self.assertTrue(worker.stand(82))
        self.input.toggle_key.assert_called_once_with(82)

    def test_happy_path_exactly_two_taps(self) -> None:
        vitals = _ScriptedVitals(
            [SIT_LOW_SP_RATIO - 0.01, 0.50, SIT_RESUME_SP_RATIO, SIT_RESUME_SP_RATIO]
        )
        worker = self._worker(vitals)
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]

        def stop_after_two() -> None:
            while self.input.toggle_key.call_count < 2 and not self.ctx.is_stopped():
                self.ctx.stop_event.wait(0.01)
            self.ctx.stop_event.set()

        threading.Thread(target=stop_after_two, daemon=True).start()
        worker.run()
        presses = [c.args[0] for c in self.input.toggle_key.call_args_list if c.args[0] == 82]
        self.assertEqual(len(presses), 2, presses)
        self.assertFalse(worker._seated)

    def test_cleanup_failure_does_not_claim_standing(self) -> None:
        worker = self._worker()
        self.ctx.stop_event.set()
        worker._seated = True
        self.input.cleanup_toggle_key.return_value = False

        self.assertFalse(worker._cleanup_stand(82))
        self.assertTrue(worker._seated)

    def test_finally_does_not_extra_tap_after_stand(self) -> None:
        worker = self._worker(_ScriptedVitals([SIT_LOW_SP_RATIO - 0.01]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._sit_until_done = MagicMock(return_value="recovered")  # type: ignore[method-assign]
        worker._seated = False
        worker._recover_sp(SIT_LOW_SP_RATIO - 0.01)
        self.input.toggle_key.assert_not_called()
        self.assertFalse(self.ctx.sitting_event.is_set())

    def test_stand_waits_600ms_before_recovery_finishes(self) -> None:
        worker = self._worker(_ScriptedVitals([SIT_RESUME_SP_RATIO, SIT_RESUME_SP_RATIO]))
        events: list[str] = []
        self.ctx.wait_unless_stopped = lambda timeout: (
            events.append(f"wait:{timeout}"), True
        )[1]  # type: ignore[method-assign]
        self.ctx.end_sit_ops = lambda: events.append("end")  # type: ignore[method-assign]
        worker._seated = True
        self.assertEqual(worker._sit_until_done(82), "recovered")
        self.assertIn(f"wait:{SIT_STAND_RESUME_DELAY_S}", events)
        self.assertNotIn("end", events)

    def test_end_sit_ops_wakes_discovery_after_generation_reset(self) -> None:
        self.ctx.discovery_wake.clear()
        self.ctx.begin_sit_ops()
        self.ctx.end_sit_ops()
        self.assertTrue(self.ctx.discovery_wake.is_set())
        self.assertFalse(self.ctx.sitting_event.is_set())
        # The recovery landing is trusted clear, so startup buffs/timers are
        # not held back by the first post-stand discovery scan.
        self.assertTrue(self.ctx.startup_area_clear.is_set())

    def test_low_sp_recovery_teleports_once_then_sits(self) -> None:
        worker = self._worker(_ScriptedVitals([0.02, 0.99]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._sit_until_done = MagicMock(return_value="recovered")  # type: ignore[method-assign]
        self.teleport.teleport_once_for_sit = MagicMock(return_value=True)  # type: ignore[method-assign]

        worker._recover_sp(0.02)

        self.teleport.teleport_once_for_sit.assert_called_once_with(log_tag="SIT")
        self.teleport.teleport_until_quiet.assert_not_called()
        worker._sit_until_done.assert_called_once_with(82)
        self.assertFalse(self.ctx.sitting_event.is_set())

    def test_post_teleport_settle_before_first_sit(self) -> None:
        """The sit toggle waits for the landing after the placement teleport."""
        worker = self._worker(_ScriptedVitals([0.02, 0.99]))
        waits: list[float] = []
        self.ctx.wait_unless_stopped = lambda t: waits.append(t) or True  # type: ignore[method-assign]
        worker._sit_until_done = MagicMock(return_value="recovered")  # type: ignore[method-assign]
        self.teleport.teleport_once_for_sit = MagicMock(return_value=True)  # type: ignore[method-assign]

        worker._recover_sp(0.02)

        self.assertEqual(waits.count(SIT_POST_TELEPORT_SETTLE_S), 1)

    def test_post_escape_settle_before_resitting(self) -> None:
        """After a danger escape the re-sit waits for the new landing too."""
        worker = self._worker(_ScriptedVitals([0.02, 0.02, 0.99, 0.99]))
        waits: list[float] = []
        self.ctx.wait_unless_stopped = lambda t: waits.append(t) or True  # type: ignore[method-assign]

        def recovery_side_effect(_scan: int) -> str:
            if worker._sit_until_done.call_count == 1:
                return "danger_escaped"
            return "recovered"

        worker._sit_until_done = MagicMock(  # type: ignore[method-assign]
            side_effect=recovery_side_effect
        )
        self.teleport.teleport_once_for_sit = MagicMock(return_value=True)  # type: ignore[method-assign]

        worker._recover_sp(0.02)

        # Once after the placement teleport, once after the danger escape.
        self.assertGreaterEqual(waits.count(SIT_POST_TELEPORT_SETTLE_S), 2)
        self.assertEqual(worker._sit_until_done.call_count, 2)
        self.assertFalse(self.ctx.sitting_event.is_set())

    def test_danger_request_survives_competing_storage_session(self) -> None:
        worker = self._worker(_ScriptedVitals([]))
        self.assertTrue(self.ctx.begin_storage_ops())
        self.ctx.request_danger_sit()

        # The sit worker cannot acquire ownership while storage is active;
        # attempting recovery must leave the queued danger request intact.
        worker._recover_sp(1.0, reason="danger")
        self.assertTrue(self.ctx.danger_sit_requested.is_set())

        self.ctx.end_storage_ops()
        self.assertTrue(self.ctx.pop_danger_sit_request())
        self.assertFalse(self.ctx.danger_sit_requested.is_set())

    def test_failed_sit_session_retries_with_gate_held(self) -> None:
        vitals = _ScriptedVitals([0.02] * 50)
        worker = self._worker(vitals)
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        attempts = {"n": 0}
        gate_during: list[bool] = []

        def sit_fail(_scan: int) -> str | None:
            attempts["n"] += 1
            gate_during.append(self.ctx.sitting_event.is_set())
            if attempts["n"] >= 3:
                self.ctx.stop_event.set()
            return None

        worker._sit_until_done = sit_fail  # type: ignore[method-assign]
        worker._recover_sp(0.02)
        self.assertGreaterEqual(attempts["n"], 3)
        self.assertTrue(all(gate_during))

    def test_damage_interrupt_stands_once(self) -> None:
        worker = self._worker(_ScriptedVitals([0.40] * 20))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.ctx.pop_danger_sit_request = MagicMock(side_effect=[False, True])  # type: ignore[method-assign]
        self.assertEqual(worker._sit_until_done(82), "danger_escaped")
        # Only enter_sit is sent. Damage escape must not press stand.
        self.assertEqual(self.input.toggle_key.call_count, 1)
        self.assertFalse(worker._seated)

    def test_damage_during_area_clear_rejects_sit_spot(self) -> None:
        worker = self._worker(_ScriptedVitals([0.40] * 20))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.ctx.request_danger_sit()
        self.teleport.danger_teleport = MagicMock(return_value=True)  # type: ignore[method-assign]

        self.assertEqual(worker._sit_until_done(82), "interrupted")
        self.input.toggle_key.assert_not_called()
        self.teleport.danger_teleport.assert_called_once_with(
            reason="sit_spot_danger",
            prefer_safe_key=True,
        )
        self.assertFalse(worker._seated)

    def test_damage_while_seated_stands_before_urgent_teleport(self) -> None:
        worker = self._worker(_ScriptedVitals([0.40] * 20))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.ctx.pop_danger_sit_request = MagicMock(side_effect=[False, True])  # type: ignore[method-assign]
        self.teleport.danger_teleport = MagicMock(return_value=True)  # type: ignore[method-assign]
        worker._seated = False

        self.assertEqual(worker._sit_until_done(82), "danger_escaped")
        self.assertEqual(
            [call.args[0] for call in self.input.toggle_key.call_args_list],
            [82],
        )
        self.teleport.danger_teleport.assert_called_once_with(
            reason="sit_danger",
            prefer_safe_key=True,
        )
        self.assertFalse(worker._seated)

    def test_unreadable_hp_then_damage_while_seated_triggers_teleport(self) -> None:
        """A transient HP read gap must not swallow seated damage escape."""
        vitals = PlayerVitals()
        danger = DangerDetector(self.ctx, vitals=vitals)
        self.ctx.danger_detector = danger
        worker = self._worker(vitals)
        worker._danger = danger
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.input.toggle_key.return_value = True
        self.input.teleport_key.return_value = True

        vitals.publish_hp(90, 100)
        danger._poll_hp()
        original_sp_pair = vitals.sp_pair
        damage_injected = False

        def sp_pair_with_damage() -> tuple[int | None, int | None]:
            nonlocal damage_injected
            if not damage_injected:
                damage_injected = True
                # The first SP read occurs after the worker has pressed sit.
                # Mark the seated session as the owner before simulating an
                # unreadable HP sample followed by real damage.
                self.ctx.sitting_event.set()
                vitals.publish_hp(None, None)
                danger._poll_hp()
                vitals.publish_hp(80, 100)
                danger._poll_hp()
            return original_sp_pair()

        vitals.sp_pair = sp_pair_with_damage  # type: ignore[method-assign]

        self.assertEqual(worker._sit_until_done(82), "danger_escaped")
        # The character sits once, then damage causes a direct teleport with
        # no stand/second-toggle input. The seated escape uses the safe
        # creamy / save-point key (17), not the random fly wing (16).
        self.input.toggle_key.assert_called_once_with(82)
        self.input.teleport_key.assert_called_once_with(17)
        self.assertFalse(worker._seated)

    def test_damage_preserved_during_escape_settle_triggers_next_escape(self) -> None:
        """Settle damage suppressed from the queue must not be ignored."""
        vitals = PlayerVitals()
        danger = DangerDetector(self.ctx, vitals=vitals)
        self.ctx.danger_detector = danger
        worker = self._worker(vitals)
        worker._danger = danger
        self.input.teleport_key.return_value = True
        self.config.creamy_tp_scan_code = 17
        self.config.creamy_tp_button = "w"
        self.config.teleport_scan_code = 16
        self.config.teleport_button = "q"
        self.config.teleport_duration_ms = 10

        # Establish the first damage event while the recovery session owns the
        # seated gate. The worker consumes this request before escaping.
        vitals.publish_hp(90, 100)
        danger._poll_hp()
        self.ctx.sitting_event.set()
        vitals.publish_hp(80, 100)
        danger._poll_hp()
        self.assertTrue(self.ctx.pop_danger_sit_request())

        # Inject the second drop from the real teleport settle wait. The
        # escape gate is active at this point, so DangerDetector records the
        # damage but deliberately does not enqueue another sit request.
        settle_polls = {"count": 0}

        def settle_wait(_timeout: float) -> bool:
            if settle_polls["count"] == 0:
                settle_polls["count"] += 1
                vitals.publish_hp(40, 100)
                danger._poll_hp()
            return True

        self.ctx.wait_unless_stopped = settle_wait  # type: ignore[method-assign]
        self.assertTrue(worker._urgent_escape(reason="sit_danger"))
        self.assertEqual(settle_polls["count"], 1)
        self.assertFalse(self.ctx.danger_sit_requested.is_set())
        self.assertEqual(self.input.toggle_key.call_count, 0)

        # The seated owner must recover the preserved detector state through
        # the real sit-recovery entrypoint. It consumes the mirrored critical
        # request, performs one second escape, and never clicks sit.
        self.assertEqual(worker._sit_until_done(82), "interrupted")
        self.assertEqual(self.input.teleport_key.call_count, 2)
        self.assertFalse(self.ctx.critical_danger_requested.is_set())
        self.assertEqual(danger.danger_level(), DangerLevel.SAFE)
        self.assertFalse(worker._sit_danger_detected())

    def test_damage_after_resit_triggers_second_danger_escape(self) -> None:
        """A fresh hit after re-sitting must not be treated as stale."""
        vitals = PlayerVitals()
        danger = DangerDetector(self.ctx, vitals=vitals)
        self.ctx.danger_detector = danger
        worker = self._worker(vitals)
        worker._danger = danger
        self.input.toggle_key.return_value = True
        self.input.teleport_key.return_value = True

        # Establish the detector baseline before the recovery sequence. Each
        # accepted sit toggle then represents a real seated interval: damage is
        # injected during the sit-key settle after the character has entered
        # the seated pose. The second hit occurs only after the recovery has
        # escaped and re-sat in the new area.
        vitals.publish_hp(100, 100)
        danger._poll_hp()
        toggle_count = {"count": 0}
        second_damage_injected = {"value": False}

        def toggle(_scan: int) -> bool:
            toggle_count["count"] += 1
            return True

        self.input.toggle_key.side_effect = toggle

        def wait_unless_stopped(timeout: float) -> bool:
            if timeout == SIT_KEY_SETTLE_S and toggle_count["count"] == 1:
                vitals.publish_hp(80, 100)
                danger._poll_hp()
            elif timeout == SIT_KEY_SETTLE_S and toggle_count["count"] == 2:
                second_damage_injected["value"] = True
                vitals.publish_hp(40, 100)
                danger._poll_hp()
            return True

        self.ctx.wait_unless_stopped = wait_unless_stopped  # type: ignore[method-assign]
        danger_teleports = {"count": 0}
        real_danger_teleport = self.teleport.danger_teleport

        def danger_escape(*, reason: str, prefer_safe_key: bool = False) -> bool:
            danger_teleports["count"] += 1
            escaped = real_danger_teleport(
                reason=reason,
                prefer_safe_key=prefer_safe_key,
            )
            if danger_teleports["count"] == 2:
                # Stop after the second escape has completed. This keeps the
                # assertion focused on the reported sequence and avoids a
                # third sit attempt during teardown.
                self.ctx.stop_event.set()
            return escaped

        self.teleport.danger_teleport = MagicMock(  # type: ignore[method-assign]
            side_effect=danger_escape
        )

        worker._recover_sp(0.02)

        # One placement teleport, then one danger teleport for each seated hit.
        self.assertEqual(self.teleport.danger_teleport.call_count, 2)
        self.assertEqual(self.input.teleport_key.call_count, 3)
        self.assertEqual(toggle_count["count"], 2)
        self.assertEqual(self.input.toggle_key.call_count, 2)
        self.assertTrue(second_damage_injected["value"])
        self.assertEqual(
            [call.kwargs["prefer_safe_key"] for call in self.teleport.danger_teleport.call_args_list],
            [True, True],
        )
        self.assertFalse(worker._seated)
        self.assertEqual(danger.danger_level(), DangerLevel.SAFE)
        self.assertFalse(self.ctx.critical_danger_requested.is_set())
        self.assertFalse(self.ctx.danger_sit_requested.is_set())

    def test_critical_hunting_escape_retries_after_failed_teleport(self) -> None:
        self.danger.danger_level.return_value = DangerLevel.CRITICAL
        self.ctx.request_critical_danger()
        self.teleport.danger_teleport = MagicMock(  # type: ignore[method-assign]
            side_effect=[False, True]
        )
        gameplay = GameplayLoop(
            self.ctx,
            attack=MagicMock(),
            teleport=self.teleport,
            input_backend=self.input,
        )

        self.assertFalse(gameplay._process_critical_danger())
        self.assertTrue(self.ctx.critical_danger_requested.is_set())
        self.assertTrue(gameplay._process_critical_danger())

        self.assertEqual(self.teleport.danger_teleport.call_count, 2)
        self.assertFalse(self.ctx.critical_danger_requested.is_set())

    def test_critical_escape_does_not_steal_active_sit_session(self) -> None:
        """SP-sit recovery owns seated danger; critical must not preempt it."""
        self.danger.danger_level.return_value = DangerLevel.CRITICAL
        self.ctx.request_critical_danger()
        self.ctx.sitting_event.set()
        self.teleport.danger_teleport = MagicMock(return_value=True)  # type: ignore[method-assign]
        gameplay = GameplayLoop(
            self.ctx,
            attack=MagicMock(),
            teleport=self.teleport,
            input_backend=self.input,
        )

        self.assertFalse(gameplay._process_critical_danger())
        self.teleport.danger_teleport.assert_not_called()
        self.assertTrue(self.ctx.critical_danger_requested.is_set())
        self.assertTrue(self.ctx.sitting_event.is_set())
        self.assertFalse(self.ctx.danger_escape_active.is_set())

    def test_ok_to_act_yields_to_pending_critical_escape(self) -> None:
        worker = self._worker()
        self.ctx.request_critical_danger()
        self.assertFalse(worker._ok_to_act())
        self.ctx.pop_critical_danger()
        self.assertFalse(self.ctx.critical_danger_requested.is_set())
        self.assertTrue(worker._ok_to_act())

    def test_tap_skips_input_while_escape_in_flight(self) -> None:
        worker = self._worker()
        self.ctx.danger_escape_active.set()
        self.assertFalse(worker._tap(82, why="enter_sit"))
        self.input.toggle_key.assert_not_called()

    def test_leftover_critical_is_consumed_not_waited(self) -> None:
        """Regression: a critical request latched before the sit claim must be
        consumed by the seated owner, never waited on. Waiting used to park the
        gameplay thread forever (the critical worker yields to a true SP sit).
        """
        worker = self._worker(_ScriptedVitals([0.99, 0.99]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.teleport.teleport_once_for_sit = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.teleport.danger_teleport = MagicMock(return_value=True)  # type: ignore[method-assign]
        # Damage is still live when the leftover is consumed: the session must
        # escape once with the safe key, then continue recovery to completion.
        levels = [DangerLevel.CRITICAL]
        self.danger.danger_level.side_effect = (
            lambda: levels.pop(0) if levels else DangerLevel.SAFE
        )
        worker._sit_until_done = MagicMock(  # type: ignore[method-assign]
            return_value="recovered"
        )
        self.ctx.request_critical_danger()

        worker._recover_sp(0.02)

        self.assertFalse(self.ctx.critical_danger_requested.is_set())
        self.teleport.danger_teleport.assert_called_once_with(
            reason="sit_danger_request",
            prefer_safe_key=True,
        )
        self.assertFalse(self.ctx.sitting_event.is_set())
        self.assertFalse(worker._seated)
        worker._sit_until_done.assert_called_once_with(82)

    def test_stale_leftover_critical_is_consumed_without_escape(self) -> None:
        """A critical request whose damage has aged out is dropped by the
        seated owner without wasting a teleport."""
        worker = self._worker(_ScriptedVitals([0.99, 0.99]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.teleport.danger_teleport = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.ctx.request_critical_danger()

        # SP is already high, so the recovery path consumes the leftover and
        # completes without any escape.
        worker._recover_sp(0.02)

        self.assertFalse(self.ctx.critical_danger_requested.is_set())
        self.teleport.danger_teleport.assert_not_called()
        self.assertFalse(self.ctx.sitting_event.is_set())
        self.assertFalse(worker._seated)

    def test_finish_recovery_consumes_leftover_critical_and_stands(self) -> None:
        """Teardown must consume a leftover critical instead of blocking the
        stand forever on an escape that can never preempt this session.
        """
        worker = self._worker()
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.assertTrue(self.ctx.try_begin_sit_ops())
        worker._seated = True
        self.ctx.request_critical_danger()

        gen_before = self.ctx.hunt_generation
        worker._finish_recovery_session(sit_scan=82)

        self.assertFalse(self.ctx.critical_danger_requested.is_set())
        self.input.toggle_key.assert_called_once_with(82)
        self.assertFalse(worker._seated)
        self.assertFalse(self.ctx.sitting_event.is_set())
        self.assertEqual(self.ctx.hunt_generation, gen_before + 1)

    def test_finish_recovery_normal_completion_stands_and_starts_new_hunt(self) -> None:
        """Without a preemption the teardown must still stand and begin a new
        hunt — the durable marker must not suppress the ordinary path.
        """
        worker = self._worker()
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.assertTrue(self.ctx.try_begin_sit_ops())
        worker._seated = True

        gen_before = self.ctx.hunt_generation
        worker._finish_recovery_session(sit_scan=82)

        self.input.toggle_key.assert_called_once_with(82)
        self.assertFalse(worker._seated)
        self.assertFalse(self.ctx.sitting_event.is_set())
        self.assertEqual(self.ctx.hunt_generation, gen_before + 1)

    def test_end_sit_ops_clears_stale_danger_sit_and_post_tp_heal(self) -> None:
        """Damage during stand settle must not leave combat blocked forever."""
        self.assertTrue(self.ctx.try_begin_sit_ops())
        self.ctx.request_danger_sit()
        self.ctx.mark_post_teleport_heal(2.0)
        self.assertTrue(self.ctx.danger_sit_requested.is_set())
        self.assertTrue(self.ctx.in_post_teleport_heal_window())

        self.ctx.end_sit_ops()

        self.assertFalse(self.ctx.danger_sit_requested.is_set())
        self.assertFalse(self.ctx.in_post_teleport_heal_window())
        self.assertFalse(self.ctx.sitting_event.is_set())

    def test_post_teleport_heal_window_is_time_bounded(self) -> None:
        """A blind OCR feed must not park gameplay on the full-HP gate forever."""
        from pybot.runtime import gate_controller

        with patch.object(gate_controller.time, "monotonic", return_value=100.0):
            self.ctx.mark_post_teleport_heal(2.0)
            self.assertTrue(self.ctx.in_post_teleport_heal_window())
        with patch.object(gate_controller.time, "monotonic", return_value=103.0):
            self.assertFalse(self.ctx.in_post_teleport_heal_window())
        # The expired gate never re-opens on later reads, and a fresh teleport
        # re-arms it.
        with patch.object(gate_controller.time, "monotonic", return_value=200.0):
            self.assertFalse(self.ctx.in_post_teleport_heal_window())
            self.ctx.mark_post_teleport_heal(2.0)
            self.assertTrue(self.ctx.in_post_teleport_heal_window())


    def test_process_pending_drops_stale_danger_sit_when_sp_ok(self) -> None:
        worker = self._worker(_ScriptedVitals([0.99]))
        self.ctx.request_danger_sit()
        self.assertFalse(worker.process_pending())
        self.assertFalse(self.ctx.danger_sit_requested.is_set())

    def test_critical_escape_waits_while_paused_without_repeating_teleport(self) -> None:
        self.danger.danger_level.return_value = DangerLevel.CRITICAL
        self.ctx.request_critical_danger()
        self.ctx.mark_paused()
        self.teleport.danger_teleport = MagicMock(return_value=True)  # type: ignore[method-assign]
        gameplay = GameplayLoop(
            self.ctx,
            attack=MagicMock(),
            teleport=self.teleport,
            input_backend=self.input,
        )

        self.assertFalse(gameplay._process_critical_danger())
        self.assertFalse(gameplay._process_critical_danger())
        self.teleport.danger_teleport.assert_not_called()
        self.assertTrue(self.ctx.critical_danger_requested.is_set())

        self.ctx.mark_running()
        self.assertTrue(gameplay._process_critical_danger())
        self.teleport.danger_teleport.assert_called_once_with(reason="critical_hunt")
        self.assertFalse(self.ctx.critical_danger_requested.is_set())

    def test_critical_escape_does_not_send_input_after_stop(self) -> None:
        self.ctx.request_critical_danger()
        self.ctx.stop_event.set()
        self.teleport.danger_teleport = MagicMock(return_value=True)  # type: ignore[method-assign]
        gameplay = GameplayLoop(
            self.ctx,
            attack=MagicMock(),
            teleport=self.teleport,
            input_backend=self.input,
        )

        self.assertFalse(gameplay._process_critical_danger())
        self.teleport.danger_teleport.assert_not_called()
        self.assertTrue(self.ctx.critical_danger_requested.is_set())

    def test_damage_during_sp_recovery_teleports_then_sits_again(self) -> None:
        worker = self._worker(_ScriptedVitals([0.02, 0.02, 0.99, 0.99]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        def recovery_side_effect(_scan: int) -> str:
            if worker._sit_until_done.call_count == 1:
                worker._seated = False
                return "danger_escaped"
            return "recovered"

        worker._sit_until_done = MagicMock(  # type: ignore[method-assign]
            side_effect=recovery_side_effect
        )
        self.teleport.teleport_once_for_sit = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.teleport.danger_teleport = MagicMock(return_value=True)  # type: ignore[method-assign]
        worker._seated = True

        worker._recover_sp(0.02)

        self.teleport.danger_teleport.assert_not_called()
        worker._sit_until_done.assert_has_calls([
            unittest.mock.call(82),
            unittest.mock.call(82),
        ])
        self.assertEqual(worker._sit_until_done.call_count, 2)
        self.assertFalse(self.ctx.sitting_event.is_set())

    def test_failed_danger_escape_retries_without_stand_toggle(self) -> None:
        worker = self._worker(_ScriptedVitals([0.40] * 20))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.ctx.pop_danger_sit_request = MagicMock(  # type: ignore[method-assign]
            side_effect=[False, True]
        )
        worker._seated = True
        self.teleport.danger_teleport = MagicMock(side_effect=[False, True])  # type: ignore[method-assign]

        self.assertEqual(worker._sit_until_done(82), "danger_escape_failed")
        self.assertEqual(self.teleport.danger_teleport.call_count, 1)
        self.assertEqual(self.input.toggle_key.call_count, 0)
        self.assertTrue(worker._seated)

        # The recovery loop owns the retry and still must not press stand.
        self.ctx.pop_danger_sit_request.side_effect = None
        self.ctx.pop_danger_sit_request.return_value = False
        worker._recover_sp(0.40, reason="danger")
        self.assertEqual(self.teleport.danger_teleport.call_count, 2)
        # The retry never sends a stand toggle while seated. The resumed
        # recovery then performs its normal sit/stand pair exactly once each.
        self.assertEqual(self.input.toggle_key.call_count, 2)
        sit_logs = [
            call.args[0]
            for call in self.ctx.logger.behavior.call_args_list
            if call.args and "tap sit-key" in call.args[0]
        ]
        self.assertEqual(
            sit_logs,
            ["[SIT] tap sit-key reason=enter_sit", "[SIT] tap sit-key reason=leave_sit"],
        )
        self.assertFalse(worker._seated)

    def test_danger_request_escapes_once_then_searches_safe_place(self) -> None:
        worker = self._worker(_ScriptedVitals([0.99, 0.99]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.ctx.request_danger_sit()
        self.ctx.request_critical_danger()
        worker._sit_until_done = MagicMock(return_value="recovered")  # type: ignore[method-assign]
        self.teleport.danger_teleport = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.teleport.teleport_once_for_sit = MagicMock(return_value=True)  # type: ignore[method-assign]

        worker._recover_sp(0.02, reason="danger")

        self.teleport.danger_teleport.assert_called_once_with(
            reason="sit_danger_request",
            prefer_safe_key=True,
        )
        self.teleport.teleport_once_for_sit.assert_not_called()
        self.assertFalse(self.ctx.danger_sit_requested.is_set())
        self.assertFalse(self.ctx.critical_danger_requested.is_set())

    def test_failed_seated_escape_retry_resits_once_in_same_session(self) -> None:
        """A failed-then-successful seated escape must not restart recovery."""
        worker = self._worker(_ScriptedVitals([0.40, 0.40, 0.99, 0.99]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._seated = True

        def resume_in_same_session(scan: int) -> str:
            self.assertTrue(worker.sit(scan))
            # The scripted helper represents a completed sit/stand recovery.
            worker._seated = False
            return "recovered"

        worker._sit_until_done = MagicMock(side_effect=resume_in_same_session)  # type: ignore[method-assign]
        self.teleport.danger_teleport = MagicMock(side_effect=[False, True])  # type: ignore[method-assign]

        # ``reason=\"danger\"`` models the already-owned seated danger
        # session. The first direct escape fails; the seated retry succeeds.
        worker._recover_sp(0.02, reason="danger")

        self.assertEqual(self.teleport.danger_teleport.call_count, 2)
        self.assertEqual(worker._sit_until_done.call_count, 1)
        # The retry stays in this recovery session and reuses the new area;
        # exactly one re-sit is emitted, not a second recovery session.
        self.assertEqual(self.input.toggle_key.call_count, 1)
        self.assertFalse(self.ctx.sitting_event.is_set())

    def test_failed_urgent_escape_retries_before_sitting_without_extra_teleport(self) -> None:
        worker = self._worker(_ScriptedVitals([0.02, 0.02, 0.99, 0.99]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.ctx.request_danger_sit()
        worker._sit_until_done = MagicMock(return_value="recovered")  # type: ignore[method-assign]
        self.teleport.danger_teleport = MagicMock(side_effect=[False, True])  # type: ignore[method-assign]
        self.teleport.teleport_once_for_sit = MagicMock(return_value=True)  # type: ignore[method-assign]

        worker._recover_sp(0.02, reason="danger")

        self.assertEqual(self.teleport.danger_teleport.call_count, 2)
        self.teleport.teleport_once_for_sit.assert_not_called()
        self.assertFalse(self.ctx.danger_sit_requested.is_set())

    def test_damage_during_escape_settle_is_consumed_once(self) -> None:
        worker = self._worker(_ScriptedVitals([0.02, 0.02, 0.99, 0.99]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.ctx.request_danger_sit()
        worker._sit_until_done = MagicMock(return_value="recovered")  # type: ignore[method-assign]

        def escape_and_report_damage(
            *,
            reason: str,
            prefer_safe_key: bool = False,
        ) -> bool:
            if self.teleport.danger_teleport.call_count == 1:
                # Simulate a new HP drop observed while the first escape is
                # settling. It must cause one new escape, not an infinite loop.
                self.ctx.request_danger_sit()
            return True

        self.teleport.danger_teleport = MagicMock(  # type: ignore[method-assign]
            side_effect=escape_and_report_damage
        )

        worker._recover_sp(0.02, reason="danger")

        self.assertEqual(self.teleport.danger_teleport.call_count, 2)
        self.assertEqual(
            [call.kwargs["reason"] for call in self.teleport.danger_teleport.call_args_list],
            ["sit_danger_request", "sit_danger_request"],
        )
        self.assertFalse(self.ctx.danger_sit_requested.is_set())
        worker._sit_until_done.assert_called_once_with(82)

    def test_damage_recovery_sits_in_urgent_escape_area_without_extra_teleport(self) -> None:
        worker = self._worker(_ScriptedVitals([0.02, 0.02, 0.99, 0.99]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._sit_until_done = MagicMock(  # type: ignore[method-assign]
            side_effect=["interrupted", "recovered"]
        )
        self.teleport.teleport_once_for_sit = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.teleport.danger_teleport = MagicMock(return_value=True)  # type: ignore[method-assign]
        worker._seated = False
        self.ctx.request_danger_sit()

        worker._recover_sp(0.02, reason="danger")

        self.teleport.danger_teleport.assert_called_once_with(
            reason="sit_danger_request",
            prefer_safe_key=True,
        )
        self.teleport.teleport_once_for_sit.assert_not_called()
        self.assertEqual(worker._sit_until_done.call_count, 2)
        self.assertFalse(self.ctx.sitting_event.is_set())

    def test_interrupted_recovery_does_not_relocate_again_before_sitting(self) -> None:
        # An interrupted attempt represents a completed urgent escape; the
        # worker should use that new area instead of teleporting again.
        worker = self._worker(_ScriptedVitals([0.02, 0.02, 0.99, 0.99]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._sit_until_done = MagicMock(  # type: ignore[method-assign]
            side_effect=["interrupted", "recovered"]
        )
        self.teleport.teleport_once_for_sit = MagicMock(  # type: ignore[method-assign]
            side_effect=[True, True]
        )
        self.teleport.danger_teleport = MagicMock(return_value=True)  # type: ignore[method-assign]
        worker._seated = False

        worker._recover_sp(0.02)

        self.assertEqual(worker._sit_until_done.call_count, 2)
        self.assertEqual(self.teleport.teleport_once_for_sit.call_count, 1)
        self.teleport.danger_teleport.assert_not_called()
        self.assertFalse(self.ctx.sitting_event.is_set())

    def test_recovered_requires_sp_still_high(self) -> None:
        worker = self._worker(_ScriptedVitals([0.99, 0.02]))
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        worker._seated = True
        worker.sit = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.ctx.pop_danger_sit_request = MagicMock(return_value=False)  # type: ignore[method-assign]
        outcome = worker._sit_until_done(82)
        self.assertIsNone(outcome)

    def test_low_unchanged_sp_does_not_retoggle_after_accepted_sit(self) -> None:
        """A historical SP change clock cannot justify a second toggle.

        The feed is alive and SP remains readable, but its numeric value stays
        below the resume threshold. The worker must keep the accepted seated
        state instead of treating the old ``changed_ms`` as proof that sitting
        failed and toggling the character back to standing.
        """
        class _ReadableUnchangedVitals(PlayerVitals):
            def sp_pair(self) -> tuple[int | None, int | None]:
                return 50, 100

            @property
            def observed_ms(self) -> int:
                return int(time.monotonic() * 1000)

            @property
            def changed_ms(self) -> int:
                return 1

        worker = self._worker(_ReadableUnchangedVitals())
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.input.toggle_key.return_value = True
        polls = {"count": 0}

        def wait_for_poll(timeout: float) -> bool:
            polls["count"] += 1
            if polls["count"] >= 2:
                self.ctx.stop_event.set()
            return False

        self.ctx.stop_event.wait = wait_for_poll  # type: ignore[method-assign]
        self.assertIsNone(worker._sit_until_done(82))
        self.assertEqual(self.input.toggle_key.call_count, 1)
        self.assertTrue(worker._seated)

    def test_stale_readable_sp_relocates_without_retoggle(self) -> None:
        """A stale last value is feed loss, not proof that sitting failed.

        The worker may relocate after the observation clock expires, but it
        must not press a second toggle first.
        """
        class _StaleReadableVitals(PlayerVitals):
            def sp_pair(self) -> tuple[int | None, int | None]:
                return 50, 100

            @property
            def observed_ms(self) -> int:
                return 1

            @property
            def changed_ms(self) -> int:
                return 1

        worker = self._worker(_StaleReadableVitals())
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.input.toggle_key.return_value = True
        self.teleport.danger_teleport = MagicMock(return_value=True)  # type: ignore[method-assign]

        clock = [1000.0]

        def fake_wait(timeout: float) -> bool:
            clock[0] += timeout
            return False

        with patch(
            "pybot.runtime.workers.sit_on_low_sp_worker.time.monotonic",
            side_effect=lambda: clock[0],
        ):
            self.ctx.stop_event.wait = fake_wait  # type: ignore[method-assign]
            outcome = worker._sit_until_done(82)

        self.assertEqual(outcome, "feed_lost")
        self.assertEqual(self.input.toggle_key.call_count, 1)
        self.teleport.danger_teleport.assert_called_once_with(
            reason="sit_feed_lost",
            prefer_safe_key=True,
        )
        self.assertFalse(worker._seated)

    def test_feed_blind_relocates_after_bound(self) -> None:
        """When SP stays unreadable (OCR layout lost / panel gone) while
        seated, recovery relocates after a bound instead of parking forever on
        a dead feed."""
        class _BlindVitals(PlayerVitals):
            def sp_pair(self) -> tuple[int | None, int | None]:
                return None, None

            @property
            def observed_ms(self) -> int:
                return int(time.monotonic() * 1000)

            @property
            def changed_ms(self) -> int:
                return int(time.monotonic() * 1000)

        vitals = _BlindVitals()
        worker = self._worker(vitals)
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.teleport.danger_teleport = MagicMock(return_value=True)  # type: ignore[method-assign]

        clock = [1000.0]

        def fake_wait(timeout: float) -> bool:
            clock[0] += timeout
            return False

        with patch(
            "pybot.runtime.workers.sit_on_low_sp_worker.time.monotonic",
            side_effect=lambda: clock[0],
        ):
            self.ctx.stop_event.wait = fake_wait  # type: ignore[method-assign]
            outcome = worker._sit_until_done(82)

        self.assertEqual(outcome, "feed_lost")
        self.teleport.danger_teleport.assert_called_once_with(
            reason="sit_feed_lost",
            prefer_safe_key=True,
        )
        self.assertFalse(worker._seated)

    def test_spot_failure_relocations_are_bounded(self) -> None:
        """Repeated blind-feed or regen-stalled spot failures relocate to a
        fresh area up to a per-session budget, then end the session cleanly —
        the sit gate is never held forever by an unrecoverable spot."""
        for outcome_name in ("feed_lost", "regen_stalled"):
            with self.subTest(outcome=outcome_name):
                worker = self._worker(
                    _ScriptedVitals([0.02, 0.02, 0.02, 0.02, 0.99, 0.99])
                )
                self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
                self.teleport.teleport_once_for_sit = MagicMock(return_value=True)  # type: ignore[method-assign]
                self.teleport.danger_teleport = MagicMock(return_value=True)  # type: ignore[method-assign]

                def spot_failure(_scan: int) -> str:
                    # Mirrors the real _sit_until_done: the feed failure
                    # escapes to a fresh area before reporting the outcome.
                    worker._urgent_escape(reason=f"sit_{outcome_name}")
                    worker._seated = False
                    return outcome_name

                worker._sit_until_done = MagicMock(  # type: ignore[method-assign]
                    side_effect=spot_failure
                )

                gen_before = self.ctx.hunt_generation
                worker._recover_sp(0.02)

                # Budget is SIT_MAX_SPOT_RELOCATIONS (3): four spot failures
                # end the session. The placement teleport happens once; each
                # spot failure escapes to a fresh area.
                self.assertEqual(worker._sit_until_done.call_count, 4)
                self.teleport.teleport_once_for_sit.assert_called_once_with(log_tag="SIT")
                self.assertEqual(self.teleport.danger_teleport.call_count, 4)
                # Teardown releases the gate cleanly: the character is
                # standing after the final spot-failure escape, so no stand
                # toggle is needed.
                self.input.toggle_key.assert_not_called()
                self.assertFalse(self.ctx.sitting_event.is_set())
                self.assertEqual(self.ctx.hunt_generation, gen_before + 1)
                self.assertFalse(worker._seated)

    def test_flat_sp_while_seated_relocates_without_retoggle(self) -> None:
        """A readable SP value frozen for a full relocation window means regen
        is blocked (re-sit toggle eaten / weight penalty). The worker relocates
        with an escape teleport — never a corrective toggle — and re-sits in
        the fresh area, instead of waiting forever on a dead spot.
        """
        class _FlatVitals(PlayerVitals):
            def sp_pair(self) -> tuple[int | None, int | None]:
                return 50, 100

            @property
            def observed_ms(self) -> int:
                return int(time.monotonic() * 1000)

            @property
            def changed_ms(self) -> int:
                return int(time.monotonic() * 1000)

        worker = self._worker(_FlatVitals())
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.input.toggle_key.return_value = True
        self.teleport.danger_teleport = MagicMock(return_value=True)

        clock = [1000.0]

        def fake_wait(timeout: float) -> bool:
            clock[0] += timeout
            return False

        with patch(
            "pybot.runtime.workers.sit_on_low_sp_worker.time.monotonic",
            side_effect=lambda: clock[0],
        ):
            self.ctx.stop_event.wait = fake_wait  # type: ignore[method-assign]
            outcome = worker._sit_until_done(82)

        self.assertEqual(outcome, "regen_stalled")
        # Only the enter_sit toggle is ever sent — the flat state is resolved
        # by relocation, never by a second corrective toggle.
        self.assertEqual(self.input.toggle_key.call_count, 1)
        self.teleport.danger_teleport.assert_called_once_with(
            reason="sit_regen_stalled",
            prefer_safe_key=True,
        )
        self.assertFalse(worker._seated)

    def test_regen_progress_is_logged_while_seated(self) -> None:
        """While waiting for regen the worker logs SP progress on a cadence, so
        a long recovery is visibly regenerating instead of looking frozen.
        """
        class _RegenVitals(PlayerVitals):
            def __init__(self) -> None:
                super().__init__()
                self._calls = 0

            def sp_pair(self) -> tuple[int | None, int | None]:
                # SP keeps changing every poll (like a regen tick) but stays
                # below the resume threshold, so neither the stand branch nor
                # the flat watchdog may fire.
                self._calls += 1
                return 50 + (self._calls % 30), 100

            @property
            def observed_ms(self) -> int:
                return int(time.monotonic() * 1000)

            @property
            def changed_ms(self) -> int:
                return int(time.monotonic() * 1000)

        worker = self._worker(_RegenVitals())
        self.ctx.wait_unless_stopped = lambda _t: True  # type: ignore[method-assign]
        self.input.toggle_key.return_value = True
        self.teleport.danger_teleport = MagicMock(return_value=True)

        clock = [1000.0]
        iterations = {"n": 0}

        def fake_wait(timeout: float) -> bool:
            iterations["n"] += 1
            clock[0] += timeout
            # Run past the flat-SP window (15s) with room to spare to prove a
            # changing SP never relocates.
            if iterations["n"] >= 350:
                self.ctx.stop_event.set()
            return False

        with patch(
            "pybot.runtime.workers.sit_on_low_sp_worker.time.monotonic",
            side_effect=lambda: clock[0],
        ):
            self.ctx.stop_event.wait = fake_wait  # type: ignore[method-assign]
            outcome = worker._sit_until_done(82)

        self.assertIsNone(outcome)
        self.teleport.danger_teleport.assert_not_called()
        regen_logs = [
            call.args[0]
            for call in self.ctx.logger.behavior.call_args_list
            if call.args and "regen sp=" in call.args[0]
        ]
        self.assertTrue(regen_logs, "expected [SIT] regen progress logs")
        self.assertIn("ratio=", regen_logs[0])
        self.assertIn("elapsed=", regen_logs[0])


if __name__ == "__main__":
    unittest.main()
