"""Death detection loop — multi-signal confirmation.

Runs independently from the coordinate-tracking worker. Captures frames
and probes every alive track at its own cadence, reading the freshest
positions from HuntTracks (written by the coords worker).

Death properties confirmed here:
1. Track is stationary (``moving`` false) — fade / silhouette clocks only
   advance then.
2. Death silhouette wins over living at the track position (confirms
   immediately on a corpse frame).
3. Opacity decays vs living baseline (primary clock when silhouette does
   not fire).
4. Attacking the track did not spend SP (when SP is readable; accelerates
   opacity confirm).

Dead tracks are removed and kill samples are recorded. This worker is the
sole death detector. The coordinate-tracking worker never flags death.
Discovery never scores death silhouettes.
"""

from __future__ import annotations

import traceback

from pybot.config.clients import MemoryAddresses
from pybot.game_state import GameMemoryPoller
from pybot.recognition.detector.tracking.opacity_probe import probe_track_death
from pybot.runtime.constants import LOG_REPEAT_INTERVAL_MS, WORKER_POLL_INTERVAL_S
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.workers.worker_contexts import DeathDetectionWorkerContext


class DeathDetectionWorker:
    """Multi-signal death detector. Reads positions from HuntTracks independently."""

    # Run at most this often when there are alive tracks to probe.
    _TICK_INTERVAL_S = 0.08
    # How long after an attack we still treat "SP did not drop" as evidence.
    _SP_NO_SPEND_WINDOW_MS = 900

    def __init__(
        self,
        ctx: DeathDetectionWorkerContext,
        memory: MemoryAddresses,
        *,
        poller: GameMemoryPoller | None = None,
    ) -> None:
        self._ctx = ctx
        self._memory = memory
        self._poller = GameMemoryPoller() if poller is None else poller
        self._last_empty_frame_log_ms = 0
        # Last SP sample from the previous death tick (global).
        self._prev_sp: int | None = None
        # track_id -> last seen attack_count
        self._attack_counts: dict[int, int] = {}
        # track_id -> (pre_attack_sp, expires_tick) while no-spend is candidate
        self._sp_no_spend: dict[int, tuple[int, int]] = {}

    def run(self) -> None:
        ctx = self._ctx
        ctx.logger.behavior("[DEATH] worker started")
        while not ctx.stop_event.is_set():
            try:
                if not ctx.should_run_workers():
                    ctx.wait_while_stopped_or_paused(WORKER_POLL_INTERVAL_S)
                    continue
                if ctx.stop_event.wait(self._TICK_INTERVAL_S):
                    break
                self._tick()
            except Exception:
                ctx.logger.behavior(
                    f"[DEATH] tick error:\n{traceback.format_exc()}"
                )

    def _read_sp(self) -> int | None:
        """Current SP or None when unavailable (no panel / no memory)."""
        ctx = self._ctx
        snap = self._poller.read(ctx.config.hwnd, self._memory)
        if not snap.ok or snap.sp is None:
            return None
        return int(snap.sp)

    def _update_sp_no_spend(
        self,
        alive_tracks: list,
        *,
        now_ms: int,
        current_sp: int | None,
    ) -> None:
        """Track attacks that did not reduce SP (dead-target property).

        When ``attack_count`` rises, compare current SP to the previous tick's
        SP. If it did not drop, mark a no-spend candidate. Later ticks in the
        window clear the mark if SP eventually falls (delayed skill cost).
        """
        alive_ids = {track.id for track in alive_tracks}
        for tid in list(self._attack_counts):
            if tid not in alive_ids:
                del self._attack_counts[tid]
        for tid in list(self._sp_no_spend):
            if tid not in alive_ids:
                del self._sp_no_spend[tid]

        prev_sp = self._prev_sp
        if current_sp is not None:
            for track in alive_tracks:
                prev_attacks = self._attack_counts.get(track.id, 0)
                if track.attack_count > prev_attacks and prev_sp is not None:
                    if current_sp >= prev_sp:
                        self._sp_no_spend[track.id] = (
                            prev_sp,
                            now_ms + self._SP_NO_SPEND_WINDOW_MS,
                        )
                    else:
                        self._sp_no_spend.pop(track.id, None)
                self._attack_counts[track.id] = track.attack_count

            for tid, (pre_sp, expires) in list(self._sp_no_spend.items()):
                if now_ms > expires:
                    del self._sp_no_spend[tid]
                    continue
                if current_sp < pre_sp:
                    del self._sp_no_spend[tid]

        self._prev_sp = current_sp

    def _sp_no_spend_for_track(self, track_id: int, *, now_ms: int) -> bool:
        entry = self._sp_no_spend.get(track_id)
        if entry is None:
            return False
        _pre_sp, expires = entry
        return now_ms <= expires

    def _tick(self) -> None:
        ctx = self._ctx
        if not ctx.capture.is_valid():
            return
        roi = ctx.capture.get_hunt_roi()
        if roi is None:
            return

        now_ms = monotonic_ms()
        area_epoch, alive_tracks = ctx.tracks.tracking_frame_snapshot(now_ms)
        if not alive_tracks:
            self._attack_counts.clear()
            self._sp_no_spend.clear()
            return

        frame = ctx.capture.capture_roi(roi)
        if frame is None or frame.size == 0:
            if now_ms - self._last_empty_frame_log_ms >= LOG_REPEAT_INTERVAL_MS:
                self._last_empty_frame_log_ms = now_ms
                ctx.logger.behavior("[DEATH] capture returned empty frame")
            return

        descriptor = ctx.tracker.ensure_descriptor()
        config = ctx.tracker.detector_config()

        current_sp = self._read_sp()
        self._update_sp_no_spend(
            alive_tracks, now_ms=now_ms, current_sp=current_sp,
        )

        death_results: list[tuple[int, float, int, int, bool]] = []

        for track in alive_tracks:
            scale = track.discovery_scale if track.discovery_scale > 0 else 1.0
            # Frame is ROI-local; track coords are screen — convert.
            local_x = int(track.x) - int(roi.x)
            local_y = int(track.y) - int(roi.y)

            death_sil_hit = False
            if not track.moving:
                death_sil_hit = bool(
                    ctx.tracker.death_wins_living_at(
                        frame, local_x, local_y, scale,
                    )
                )

            sp_no_spend = self._sp_no_spend_for_track(track.id, now_ms=now_ms)

            baseline, samples, streak, dead = probe_track_death(
                frame,
                descriptor,
                x=local_x,
                y=local_y,
                scale=scale,
                opacity_baseline=track.opacity_baseline,
                opacity_baseline_samples=track.opacity_baseline_samples,
                opacity_decay_streak=track.opacity_decay_streak,
                config=config,
                moving=track.moving,
                now_tick=now_ms,
                death_silhouette_hit=death_sil_hit,
                sp_no_spend=sp_no_spend,
            )
            death_results.append((track.id, baseline, samples, streak, dead))

        dead_ids = ctx.tracks.apply_death_results(
            death_results,
            now_tick=now_ms,
            area_epoch=area_epoch,
        )
        for tid in dead_ids:
            self._attack_counts.pop(tid, None)
            self._sp_no_spend.pop(tid, None)
        if dead_ids:
            ctx.logger.behavior(
                f"[DEATH] confirmed {len(dead_ids)} dead track(s): {dead_ids}"
            )
