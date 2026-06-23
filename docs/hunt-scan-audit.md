# Hunt Scan Architecture

## Runtime model

| Layer | Interval | Entry point | Detector command |
|-------|----------|-------------|------------------|
| **Watch** | 150ms (`HUNT_WATCH_INTERVAL_MS`) | `HuntWatchTick` | `watch` — `watch_only` at track coordinates |
| **Discovery** | 1000ms (`HUNT_DISCOVERY_INTERVAL_MS`) | `HuntDiscoveryTick` | `scan` — heatmap discovery only |
| **Hunt** | ~20–50ms loop | `Hunt()` | No detect — attack, target select, teleport |

All track state flows through **HuntTracks** (single source of truth).

## Python

- **Persistent server:** `py -3 mob-recognition/cli.py serve` (JSON-lines stdin/stdout)
- **Commands:** `scan`, `watch`, `shutdown`
- **One-shot CLI:** `detect-simple` for debug/GUI only (optional `--watch-points` = watch-only mode)

`SimpleMobDetector.detect()`:
- Default → discovery (heatmap)
- `watch_only=True` + `watch_points` → watch path (no heatmap)
- Watch points cannot be combined with discovery in one call

## AHK

- **Hunt bot:** `MobRecognitionDiscoveryDetect`, `MobRecognitionWatchDetect` via persistent server
- **GUI debug search:** `MobRecognitionDetectCli` subprocess (debug bundles only)
- **Track watch apply:** `HuntTracks_ApplyWatch` (no scan increment)
- **Discovery apply:** `HuntTracks_Update` (increments scan id)
- **Engage:** `HuntTracks_IsEngageable` (discovery fresh OR watch within 400ms)

## Benchmark reference (800×800 ROI, horn)

| Path | Typical time |
|------|--------------|
| Discovery `scan` | ~1.25s |
| Watch 3 points | ~0.05s |
| Subprocess one-shot | +0.4–0.8s spawn overhead |

Run: `py -3 mob-recognition/bench_scan_paths.py`
