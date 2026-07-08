# ViiperHexBots

Fork of [HexBots](https://github.com/Georg1992/HexBots) that sends keyboard and mouse input through [VIIPER](https://github.com/Alia5/VIIPER) virtual HID devices instead of AutoHotInterception.

The bot captures the game window, detects mobs via a SPR/ACT descriptor pipeline, tracks them across frames, and attacks targets automatically.

## Prerequisites

- Windows 64-bit
- Python 3.10+
- [usbip-win2](https://github.com/vadimgrn/usbip-win2) kernel driver (one-time install + reboot)
- Go 1.26+ (only for building `viiper.exe`, not needed at runtime)

## Build

```powershell
git submodule update --init --recursive
.\build.ps1
```

This produces `VIIPER/dist/viiper.exe`.

## Run

1. Install usbip-win2 and reboot if you have not already.
2. Run `build.ps1` once to build `viiper.exe`.
3. Launch `run.bat` to start the Python GUI.

The Python bot launches `viiper.exe` directly, sets up virtual keyboard/mouse devices via the VIIPER TCP API, and sends binary input reports over device streams — no Go bridge needed.

## Tests

```powershell
.\scripts\run_all_hunt_tests.ps1
```

Runs runtime unit tests (`pybot/runtime/tests`, `pybot/app/tests`) and mob-recognition pytest suite. Recognition tests only:

```powershell
.\scripts\run_recognition_tests.ps1
```

## Layout

```
ViiperHexBots/
  run.bat / run.pyw         Launchers
  build.ps1                 Build script for viiper.exe
  config.ini                Local runtime settings
  pybot/
    app/                    Desktop GUI (tkinter)
    runtime/                Hunt engine (workers, tracks, capture, input)
    viiper/                 Pure Python VIIPER TCP client
  mob-recognition/          SPR/ACT descriptor + heatmap detection pipeline
  assets/
    mobs/                   Source SPR/ACT per mob (input)
    generated_descriptors/  Runtime descriptors, rebuilt on launch (gitignored)
    modified_mobs/          Transformed SPR/ACT mirror (gitignored)
  clients/                  Per-server profiles (memory addresses, captcha)
  scripts/                  Descriptor build, test runners, dev tools
  logs/                     Session logs and debug output (generated)
  VIIPER/                   Git submodule (virtual HID driver)
```

## Mob descriptors

Mob sprites live in `assets/mobs/<MobName>/`. On launch the bot rebuilds descriptors into `assets/generated_descriptors/<mob>/simple/descriptor.json` and lists available mobs in the UI.

Build a single mob descriptor manually:

```powershell
.\scripts\build-mob-descriptor.ps1 -Mob horn -Force
```

See `mob-recognition/README.md` for the detection pipeline and CLI commands.

## Dev tools

| Script | Purpose |
|--------|---------|
| `scripts/smoke_test.py` | Import/init check before GUI launch |
| `scripts/test_detection.py` | Live detection overlay on game window |
| `scripts/capture_detect.py` | One-shot screenshot + detect |
| `mob-recognition/bench_*.py` | Manual detection benchmarks |

## Logs

Each app launch writes to `logs/sessions/<session-id>/`. Only the latest 3 session folders are kept.

- `behavior.log` — user-facing bot timeline (shown in the GUI log box)
- `system.log` — internal diagnostics (detector timings, runtime context)
- `summary.json` — bot-run stats (scans, attacks, scale calibration)

Debug frame dumps from dev tools go to `logs/debug_saves/` and `logs/detect_debug/` (gitignored).

## Differences from HexBots

- No AutoHotInterception DLLs or Interception driver
- Requires usbip-win2 instead
- Virtual HID devices instead of routing through physical keyboard/mouse
- Pure Python VIIPER TCP client (no Go input bridge)

## Upstream

Based on [Georg1992/HexBots](https://github.com/Georg1992/HexBots).
