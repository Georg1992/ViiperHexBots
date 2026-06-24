# Hunt Architecture

## Layers

| Layer | Module | Interval | IPC | Output |
|-------|--------|----------|-----|--------|
| **Mob recognition** | `MobRecognition.ahk` | 1000ms | `scan` | `detections[]` (living only) |
| **Mob state recognition** | `MobStateRecognition.ahk` | 150ms | `state` | `trackUpdates[]` by trackId |
| **Track store** | `HuntTracks.ahk` | — | — | `ApplyDetections`, `ApplyStateUpdates` |
| **Hunt policy** | `HuntPolicy.ahk` + `BotLogic.ahk` | ~25ms | — | select, attack, teleport |

## Responsibilities

### MobRecognition
- Heatmap discovery, living targets only
- `MobRecognitionDiscoveryDetect` → `HuntTracks_ApplyDetections`
- Does not evaluate death on known tracks

### MobStateRecognition
- `MobStateRecognize` / `MobStateRecognizeAndApply`
- Input: `[{id, x, y}]` screen coordinates
- Output: `[{trackId, state, confidence, x, y}]` where state is `alive`, `dead`, or `gone`
- Python: `cmd: state` → `evaluate_track_states` / `DeathValidator`

### HuntTracks
- Single source of truth for track identity
- `HuntTracks_ApplyDetections` — match living detections to tracks or create new ones
- `HuntTracks_ApplyStateUpdates` — state recognition removes dead/gone tracks, refreshes alive positions
- `HuntTracks_ApplyAttackEvent` — per-track attack timing only

### HuntPolicy
- Round-robin target selection: lowest `attackCount` each swing, then repeat until all tracks removed
- Teleport when tracks and scan are both empty
- Does not run vision or parse detector JSON

## Timers

- `HuntDiscoveryTick` — discovery
- `HuntStateTick` — mob state (replaces old `HuntWatchTick`)
- `Hunt()` — attack loop only

## Python

- `scan` — discovery (`_evaluate_living_center`)
- `state` — track state (`evaluate_track_states` / `DeathValidator`)
