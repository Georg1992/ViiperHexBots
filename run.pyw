"""Double-click to launch ViiperHexBots Python GUI (no console window)."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from pybot.app.main_window import main

main()
