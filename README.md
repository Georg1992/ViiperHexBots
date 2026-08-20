# ViiperHexBots

Ragnarok Online hunt bot. Fork of [HexBots](https://github.com/Georg1992/HexBots) that sends keyboard and mouse input through [VIIPER](https://github.com/Alia5/VIIPER) virtual HID devices instead of AutoHotInterception.

The bot captures the game window, finds the selected mob with a SPR/ACT descriptor pipeline, tracks identities across frames, and attacks. Hunt mode is teleport or walk. The same runtime sits on low SP, restores HP, dumps to kafra when overweight, restocks fly wings, and fly-wings out of damage. Optional per-server memory reads supply SP and weight. A Win32 overlay shows track count and occupancy.

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

This produces `VIIPER/dist/viiper.exe` and installs the Python package in editable mode (`pip install -e ".[dev]"`).

Manual install only:

```powershell
pip install -e ".[dev]"
```

## Run

1. Install usbip-win2 and reboot if you have not already.
2. Run `build.ps1` once to build `viiper.exe` and install the Python package.
3. Launch `run.bat`, `run.pyw`, or the `viiperhex` console command.

The Python bot launches `viiper.exe` directly, sets up virtual keyboard/mouse devices via the VIIPER TCP API, and sends binary input reports over device streams — no Go bridge needed.

Console entry points (after editable install):

| Command | Purpose |
|---------|---------|
| `viiperhex` | Desktop GUI |
| `viiperhex-hunt` | Headless hunt runtime CLI |
| `mob-detect` | Mob recognition CLI (`build-descriptor`, `detect`, …) |

## Tests

```powershell
pytest
```

Runs runtime tests (`tests/runtime`), app tests (`tests/app`), recognition tests (`tests/recognition`), and architecture checks (`tests/architecture`). Recognition tests only:

```powershell
pytest tests/recognition
```

## Layout

```
ViiperHexBots/
  pyproject.toml            Python package metadata, deps, pytest config
  run.bat / run.pyw         Launchers
  build.ps1                 Build script for viiper.exe
  config.ini                Local runtime settings
  pybot/
    app/                    Desktop GUI (tkinter)
    runtime/                Hunt engine (workers, tracks, capture, input)
    recognition/            SPR/ACT descriptor + heatmap detection pipeline
    game_state/             Player vitals and optional process memory
    mobs/                   Mob catalog and descriptor build helpers
    config/                 Unified settings schema and INI store
    viiper/                 Pure Python VIIPER TCP client
  assets/
    mobs/                   Source SPR/ACT per mob (input)
    generated_descriptors/  Runtime descriptors, rebuilt on launch (gitignored)
  clients/                  Per-server profiles (memory addresses)
  scripts/                  Sprite asset helper (`make_mobs_big_red.py`)
  tests/                    Pytest suite, fixtures, and debug tools
    fixtures/               Shared screenshots and recognition fixture suites
    tools/                  Test/debug utilities
  logs/                     Session logs (generated)
  VIIPER/                   Git submodule (virtual HID driver)
```

## Mob descriptors

Mob sprites live in `assets/mobs/<MobName>/`. On launch the bot rebuilds descriptors into `assets/generated_descriptors/<mob>/descriptor.json` and lists available mobs in the UI.

Build a single mob descriptor manually:

```powershell
python -m pybot.recognition build-descriptor --mob horn --force
```

Use `mob-detect` for CLI examples (`mob-detect detect --mob horn --help`). Pipeline source lives in `pybot/recognition/`.

## Dev tools

| Script | Purpose |
|--------|---------|
| `python -m tests.tools.debug_vis` | Discovery pipeline fixture visualization (`_debug_vis/`) |
| `python -m pybot.recognition fixtures --mob <name>` | Run screenshot fixture suite for one mob |
| `python -m pybot.recognition detect --mob <name> --image <path> --debug` | Write detector debug dumps |

## Logs

Each hunt writes to `logs/sessions/<session-id>/`. Only the latest 3 session folders are kept.

- `behavior.log` — hunt timeline (attacks, deaths, teleports, sit/storage)
- `system.log` — app/session diagnostics (GUI start/stop, VIIPER, imports)

Detector `--debug` dumps go to `pybot/recognition/debug/detector/` (gitignored). Fixture visualization writes `_debug_vis/`.

## Differences from HexBots

- No AutoHotInterception DLLs or Interception driver
- Requires usbip-win2 instead
- Virtual HID devices instead of routing through physical keyboard/mouse
- Pure Python VIIPER TCP client (no Go input bridge)
- Hunt engine, recognition, and GUI are Python (not AutoHotkey)

## Upstream

Based on [Georg1992/HexBots](https://github.com/Georg1992/HexBots).
