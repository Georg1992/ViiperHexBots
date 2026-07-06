# ViiperHexBots

Fork of [HexBots](https://github.com/Georg1992/HexBots) that sends keyboard and mouse input through [VIIPER](https://github.com/Alia5/VIIPER) virtual HID devices instead of AutoHotInterception.

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

## Layout

```
ViiperHexBots/
  run.bat                  Python launcher
  build.ps1                Build script for viiper.exe
  config.ini               Local runtime settings (generated)
  pybot/
    app/                   Desktop GUI (tkinter)
    runtime/               Hunt runtime logic
    viiper/                Pure Python VIIPER TCP client
  clients/                 Per-server profiles (memory addresses, captcha)
  logs/                    Runtime/session logs (generated)
  scripts/                 Maintenance and descriptor-build scripts
  mob-recognition/         Python descriptor/detection pipeline
  generated_descriptors/   Runtime mob descriptors (built from SPR/ACT assets)
  VIIPER/                  Git submodule (viiper virtual HID driver)
```

## Logs

Each app launch writes to `logs/sessions/<session-id>/`.
Only the latest 3 session folders are kept.

- `behavior.log` is the user-facing bot timeline and is the source for the current GUI log box.
- `system.log` is internal diagnostics for detector timings, runtime context, process state, and debugging.
- `<bot-session-id>/summary.json` stores bot-run stats such as scans, attacks, and scale calibration.

## Differences from HexBots

- No AutoHotInterception DLLs or Interception driver
- Requires usbip-win2 instead
- Virtual HID devices instead of routing through physical keyboard/mouse
- Pure Python VIIPER TCP client (no Go input bridge)

## Upstream

Based on [Georg1992/HexBots](https://github.com/Georg1992/HexBots).
