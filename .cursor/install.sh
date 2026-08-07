#!/usr/bin/env bash
# Cloud Agent bootstrap for ViiperHexBots.
#
# ViiperHexBots is a Windows-targeted Ragnarok Online hunt bot. The GUI
# (tkinter), win32 window/memory access, and VIIPER virtual-HID runtime only
# work on Windows. On the Linux Cloud Agent VM this script prepares the
# cross-platform layers that DO run here: the Python package and the OpenCV/
# NumPy SPR/ACT recognition + detection pipeline, plus its test suite.
set -euo pipefail

cd "$(dirname "$0")/.."

# System packages. libGL/glib (needed by opencv-python) already ship in the
# base image; tkinter is the only extra apt dependency the codebase imports.
sudo apt-get update
sudo apt-get install -y --no-install-recommends python3-tk

# Install the package (editable) plus dev extras into the system interpreter so
# `python3`, `pytest`, and the console entry points (mob-detect, viiperhex-hunt,
# ...) are on PATH without activating a virtualenv. The VM is disposable, so a
# system-wide --break-system-packages install is safe here.
sudo python3 -m pip install --break-system-packages -e ".[dev]"

# Source-derived generation: build the mob SPR/ACT descriptors into
# assets/generated_descriptors/. This is idempotent — up-to-date descriptors
# are skipped and only missing/stale ones are (re)built. The recognition tests
# require these descriptors to exist.
python3 -c "from pybot.mobs.catalog import ensure_mob_assets; ensure_mob_assets()"

echo "ViiperHexBots Cloud Agent install complete."
