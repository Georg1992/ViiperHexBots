"""HID mouse constants and wire protocol for VIIPER device streams.

Wire format (client → server)::
    buttons:u8 dx:i16 dy:i16 wheel:i16 pan:i16   (9 bytes total)

    Byte 0: Button bitfield (bit 0=Left, 1=Right, 2=Middle, 3=Back, 4=Forward)
    Bytes 1-2: Delta X (int16 little-endian)
    Bytes 3-4: Delta Y (int16 little-endian)
    Bytes 5-6: Wheel (int16 little-endian)
    Bytes 7-8: Pan (int16 little-endian)

There is no server → client feedback for the mouse device.
"""

from __future__ import annotations

import struct

# ── Button bit masks ──────────────────────────────────────────────────
BTN_LEFT = 0x01
BTN_RIGHT = 0x02
BTN_MIDDLE = 0x04
BTN_BACK = 0x08
BTN_FORWARD = 0x10

# Map bridge button indices to VIIPER mouse flags:
# 0=left, 1=right, 2=middle, 3=back, 4=forward
BUTTON_INDEX_TO_FLAG: dict[int, int] = {
    0: BTN_LEFT,
    1: BTN_RIGHT,
    2: BTN_MIDDLE,
    3: BTN_BACK,
    4: BTN_FORWARD,
}


class MouseState:
    """Represents the state of a HID mouse.

    Uses relative movement (deltas) rather than absolute positioning.
    """

    __slots__ = ("buttons", "dx", "dy", "wheel", "pan")

    def __init__(
        self,
        buttons: int = 0,
        dx: int = 0,
        dy: int = 0,
        wheel: int = 0,
        pan: int = 0,
    ) -> None:
        self.buttons: int = buttons & 0x1F  # 5 buttons
        self.dx: int = dx
        self.dy: int = dy
        self.wheel: int = wheel
        self.pan: int = pan

    def marshal(self) -> bytes:
        """Encode to VIIPER wire format: 9 bytes.

        buttons:u8 dx:i16 dy:i16 wheel:i16 pan:i16
        """
        return struct.pack(
            "<Bhhhh",
            self.buttons & 0x1F,
            self.dx,
            self.dy,
            self.wheel,
            self.pan,
        )
