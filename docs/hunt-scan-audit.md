# Hunt Scan Architecture

## Runtime model

| Layer | Interval | Entry point | Detector command |
|-------|----------|-------------|------------------|
| **Watch** | 150ms (`HUNT_WATCH_INTERVAL_MS`) | `HuntWatchTick` | `watch` — `watch_only` at alive track coordinates |
| **Discovery** | 1000ms (`HUNT_DISCOVERY_INTERVAL_MS`) | `HuntDiscoveryTick` | `scan` — heatmap, always runs |
| **Hunt** | ~25ms loop | `Hunt()` | attack + target select only |

All track state flows through **HuntTracks** (single source of truth).

## Rules

1. **Discovery** runs every second regardless of combat. It creates/updates tracks for every living horn on screen.
2. **Watch** runs only while alive tracks exist. It updates positions and marks deaths at track coordinates.
3. **Attack** uses alive tracks only. One path: `HuntAttackTrack` (skill held through click).
4. **Teleport** only from `HuntDiscoveryTick` when all of:
   - `CurrentTargetTrackId` is empty
   - `HuntTracks_GetAliveCount()` is 0
   - latest discovery scan has 0 living candidates
   - area was engaged (`huntAreaEngaged`) or initial seek (`huntSeekMobWarp`) is active

No clear-scan counters, no discovery pause during combat, no teleport from watch.

## AHK

- **Discovery timer:** periodic `SetTimer` only; first scan via direct `HuntDiscoveryTick()` in `HuntStartScanTimers`.
- **Do not** follow periodic `SetTimer` with `SetTimer, -1` on the same label (AHK replaces the timer).

## Python

- Persistent server: `py -3 mob-recognition/cli.py serve --ipc-dir %TEMP%\mob_recognition_ipc`
- Commands: `scan`, `watch`, `shutdown`
