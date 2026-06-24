# Hunt Architecture

## Layers

| Layer | Module | Interval | IPC | Output |
|-------|--------|----------|-----|--------|
| **Discovery** | `MobRecognition.ahk` | 1000ms | `scan` | living detections → `HuntTracks_ApplyDetections` |
| **Mob state** | `MobStateRecognition.ahk` | 150ms | `state` | `trackUpdates[]` → `HuntTracks_ApplyStateUpdates` |
| **Direct state** | `BotLogic.ahk` + `MobStateRecognition.ahk` | post-attack +120ms | `state` mode `direct` | single-track update |
| **Track store** | `HuntTracks.ahk` | — | — | track storage, pendingResult, area epoch |
| **Hunt policy** | `HuntPolicy.ahk` | — | — | target selection, teleport gate |
| **Hunt loop** | `BotLogic.ahk` | ~25ms | — | attack, timers, DirectState queue |

## Responsibilities

### MobRecognition (Discovery)
- Heatmap discovery; living targets only
- `MobRecognitionDiscoveryDetect` → parse candidates → `HuntTracks_ApplyDetections`
- Does not evaluate death on known tracks

### MobStateRecognition
- Periodic: `MobStateRecognizeAndApply` for all alive tracks
- Direct: `MobStateRecognizeDirectAndApply` for one track after attack
- Input: `[{id, x, y}]` screen coordinates
- Output: `[{trackId, state, confidence, x, y}]` where state is `alive`, `dead`, `gone`, or `unknown` (direct only)
- Python: `cmd: state` → `evaluate_track_states` / `evaluate_track_state_direct`

### HuntTracks
- Single source of truth for track identity; presence in `huntTracks` means alive
- `HuntTracks_ApplyDetections` — match living detections or create tracks
- `HuntTracks_ApplyStateUpdates` — remove dead/gone, refresh alive, record kills
- `HuntTracks_ApplyAttackEvent` — attack timing and **pendingResult** window
- `HuntAreaReset` — clears tracks, increments **AreaEpoch**, clears DirectState queue

### HuntPolicy
- Round-robin: lowest `attackCount` each swing; skip tracks with pendingResult
- Teleport when tracks and discovery scan are both empty
- Does not run vision or parse detector JSON

### BotLogic (DirectState + scheduling)
- `huntPendingDirectState` — one queued direct check per attack (`trackId`, `x`, `y`, `areaEpoch`, `readyTick`)
- Discovery blocked only when a **ready and valid** direct request exists
- Stale direct requests dropped on epoch mismatch, missing track, or invalid ROI
- `huntServerBusy` — single Python IPC slot shared by discovery, state, and direct

## pendingResult

After `HuntTracks_ApplyAttackEvent`:
- `pendingResultUntilTick = now + HUNT_ATTACK_RESULT_WINDOW_MS` (1800ms)
- `HuntPolicy` skips the track while pending
- State `alive` clears pending via `HuntTracks_ClearPendingResult`
- Timeout re-allows attack without marking dead

## AreaEpoch

- `HUNT_AREA_EPOCH` incremented in `HuntAreaReset` (teleport area clear)
- DirectState queue stores epoch at schedule time; dropped if epoch changed before run

## Timers

- `HuntDiscoveryTick` — discovery (1000ms); runs immediately on timer start if IPC idle
- `HuntStateTick` — periodic state (150ms); direct check takes priority when ready
- `Hunt()` — attack loop only (~25ms idle sleep when no target)

## Logging prefixes

| Prefix | Source |
|--------|--------|
| `[DISCOVERY]` | discovery skip/fail |
| `[STATE]` | direct state results; periodic detail when `MOB_STATE_DEBUG` |
| `[TRACK]` | track create/remove/state apply |
| `[HUNT]` | attack, target, teleport, area clear |
| `[DIRECT]` | direct queue/run/clear/drop |

## Session stats

`BotSession` records `kills` (state `dead`), `teleports` (`Teleport()`), `attacksIssued`, scans.

## Python (unchanged algorithms)

- `scan` — discovery
- `state` — track state (`evaluate_track_states` / `evaluate_track_state_direct`)
- Detector watch-point drift keys in `config_simple.json` are internal search geometry, not AHK watch slots

## AHK timing constants

| Constant | Value | File |
|----------|-------|------|
| `HUNT_STATE_INTERVAL_MS` | 150 | BotLogic |
| `HUNT_DISCOVERY_INTERVAL_MS` | 1000 | BotLogic |
| `HUNT_POST_ATTACK_STATE_DELAY_MS` | 120 | BotLogic |
| `HUNT_TRACK_MATCH_RADIUS` | 45 | HuntTracks |
| `HUNT_ATTACK_RESULT_WINDOW_MS` | 1800 | HuntTracks |
