"""Attack loop — round-robin, teleport, shadow/live input."""

from __future__ import annotations

import time

from pybot.runtime._mob_rec_path import import_hunt_track_rules
_hunt_rules = import_hunt_track_rules()
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime import overlay as hunt_overlay
from pybot.runtime.workers.worker_contexts import AttackLoopContext


class AttackLoop:
    def __init__(
        self,
        ctx: AttackLoopContext,
        hunt_mode,
        input_backend: InputBackend,
    ) -> None:
        self._ctx = ctx
        self._hunt_mode = hunt_mode
        self._input = input_backend
        self._last_skill_attack_ms = 0
        self._last_no_attackable_log_ms = 0

    def run(self) -> None:
        self._ctx.logger.behavior("[HUNT] attack loop started")
        while not self._ctx.is_stopped():
            if not self._ctx.should_run_workers():
                self._ctx.stop_event.wait(0.05)
                continue

            tick = monotonic_ms()
            if self._skill_delay_remaining(tick) > 0:
                self._ctx.stop_event.wait(0.025)
                continue

            target_id = self._ctx.policy.select_target(
                self._ctx.tracks.tracks_for_policy(tick),
                tick,
            )
            if target_id:
                self._handle_target(target_id, tick)
                continue

            self._hunt_mode.on_no_attackable_targets()
            self._maybe_log_no_attackable(tick)
            self._ctx.stop_event.wait(0.025)

    def _skill_delay_remaining(self, now_tick: int) -> int:
        if not self._last_skill_attack_ms:
            return 0
        remaining = self._ctx.config.skill_delay_ms - (now_tick - self._last_skill_attack_ms)
        return remaining if remaining > 0 else 0

    def _maybe_log_no_attackable(self, now_tick: int) -> None:
        if now_tick - self._last_no_attackable_log_ms < 2000:
            return
        self._last_no_attackable_log_ms = now_tick
        known = self._ctx.tracks.get_alive_or_pending_count(now_tick)
        attackable = self._ctx.tracks.get_attackable_count(now_tick)
        self._ctx.logger.behavior(
            f"[HUNT] no attackable tracks known={known} attackable={attackable}"
        )

    def _handle_target(self, target_id: int, now_tick: int) -> None:


        ctx = self._ctx
        track = ctx.tracks.get_track_by_id(target_id)
        if track is None:
            ctx.validation.log_attack_decision(target_id, "track_missing")
            ctx.logger.behavior(f"[HUNT] abort attack id={target_id} reason=track_missing")
            return

        if not _hunt_rules.is_attackable(track, now_tick):
            block = "pending" if track.state == "pending" else "not_alive"
            ctx.validation.log_attack_decision(
                target_id,
                block,
                x=track.x,
                y=track.y,
                coord_age_ms=ctx.tracks.get_coord_age_ms(target_id, now_tick),
                attack_count=track.attack_count,
                state=track.state,
            )
            return

        coord_age = ctx.tracks.get_coord_age_ms(target_id, now_tick)
        stale_limit = ctx.config.coord_stale_skip_ms
        if stale_limit is not None and coord_age > stale_limit:
            ctx.urgent.schedule_direct(
                target_id,
                track.x,
                track.y,
                delay_ms=0,
                now_ms=now_tick,
            )
            ctx.validation.log_attack_decision(
                target_id,
                "stale_coords",
                x=track.x,
                y=track.y,
                coord_age_ms=coord_age,
                attack_count=track.attack_count,
                state=track.state,
            )
            ctx.logger.behavior(
                f"[HUNT] skip stale id={target_id} ageMs={coord_age} limit={stale_limit}"
            )
            return

        # Skip tracks that haven't had a state confirm recently (death detection gap)
        confirm_age_ms = now_tick - track.last_confirm_tick
        if confirm_age_ms >= _hunt_rules.HUNT_MAX_CONFIRM_AGE_MS:
            ctx.urgent.schedule_direct(
                target_id,
                track.x,
                track.y,
                delay_ms=0,
                now_ms=now_tick,
            )
            ctx.validation.log_attack_decision(
                target_id,
                "stale_confirm",
                x=track.x,
                y=track.y,
                coord_age_ms=coord_age,
                attack_count=track.attack_count,
                state=track.state,
            )
            ctx.logger.behavior(
                f"[HUNT] skip stale confirm id={target_id} confirmAgeMs={confirm_age_ms}"
            )
            return

        ctx.validation.log_attack_decision(
            target_id,
            "",
            x=track.x,
            y=track.y,
            coord_age_ms=coord_age,
            attack_count=track.attack_count,
            state=track.state,
        )

        # ── Re-read fresh coordinates right before the click ───────
        # The tracking worker may have updated the track's position
        # since we first read it at function entry.  Re-fetch so we
        # always click the freshest coordinate the tracker knows.
        # If the track was removed between the entry read and this
        # re-read (TOCTOU race with confirm worker), fall back to
        # the original coordinates rather than aborting the attack.
        fresh = ctx.tracks.get_track_by_id(target_id)
        if fresh is not None:
            click_x, click_y = fresh.x, fresh.y
            attack_count = fresh.attack_count
        else:
            ctx.logger.behavior(
                f"[HUNT] track removed before click id={target_id} using stale coords"
            )
            click_x, click_y = track.x, track.y
            attack_count = track.attack_count
        ctx.validation.log_attack_engage(
            target_id,
            click_x,
            click_y,
            coord_age_ms=ctx.tracks.get_coord_age_ms(target_id, monotonic_ms()),
            attack_count=attack_count,
        )

        self._input.move_mouse(click_x, click_y)
        self._input.skill_click(ctx.config.skill_scan_code)

        if ctx.tracks.apply_attack_event(target_id, now_tick=now_tick):
            ctx.policy.note_attack_target(target_id)
            self._last_skill_attack_ms = now_tick
            hunt_overlay.increment_attacks()
            ctx.urgent.schedule_direct(
                target_id,
                click_x,
                click_y,
                delay_ms=ctx.config.post_attack_state_delay_ms,
                now_ms=now_tick,
            )
            ctx.logger.behavior(
                f"[HUNT] attack id={target_id} pendingUntil="
                f"{now_tick + ctx.config.post_attack_state_delay_ms}"
            )

        deadline = time.monotonic() + (ctx.config.skill_delay_ms / 1000.0)
        while time.monotonic() < deadline and not ctx.is_stopped():
            ctx.stop_event.wait(0.025)
