# ViiperHexBots

Fork of [HexBots](https://github.com/Georg1992/HexBots) that sends keyboard and mouse input through [VIIPER](https://github.com/Alia5/VIIPER) virtual HID devices instead of AutoHotInterception.

## Prerequisites

- Windows 64-bit
- AutoHotkey v1.1.33+
- Go 1.26+ (for building the input bridge)
- [usbip-win2](https://github.com/vadimgrn/usbip-win2) kernel driver (one-time install + reboot)

## Build

```powershell
git submodule update --init --recursive
.\build.ps1
```

This produces `viiper-input.exe` in the project root (embeds `viiper.exe`).

## Run

1. Install usbip-win2 and reboot if you have not already.
2. Run `build.ps1` once to build `viiper-input.exe`.
3. Launch `main.ahk` with AutoHotkey v1.

On startup the script launches `viiper-input.exe`, which starts the VIIPER server and creates virtual keyboard/mouse devices.

## Layout

```
ViiperHexBots/
  main.ahk                 Bot GUI and entry point
  BotLogic.ahk             Hunting, inventory, warp logic
  utilityFunctions.ahk     Input helpers (AHI-compatible API)
  Lib/ViiperInput.ahk      AHK client for the input bridge
  input-bridge/            Go HTTP bridge to VIIPER
  viiper-input.exe         Built bridge (not in git)
  VIIPER/                  Git submodule
  build.ps1                Build script
```

## Differences from HexBots

- No AutoHotInterception DLLs or Interception driver
- No VID/PID device detection (`DeviceDetector.ahk` removed)
- Requires usbip-win2 instead
- Virtual HID devices instead of routing through physical keyboard/mouse

## Upstream

Based on [Georg1992/HexBots](https://github.com/Georg1992/HexBots).
