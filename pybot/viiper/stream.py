"""VIIPER device stream connection.

After creating a device with ``ViiperClient.device_add()``, you must
connect to its stream within the connect timeout (default 5s) to keep
it alive. The stream is a long-lived TCP connection over which you send
binary input reports (device-specific wire format) and receive binary
feedback (rumble, LEDs, etc.).
"""

from __future__ import annotations

import socket
from typing import Callable


class DeviceStream:
    """A persistent TCP stream to a VIIPER device for binary input reports.

    Usage::

        stream = DeviceStream.open("127.0.0.1:3242", bus_id=1, dev_id="1")
        stream.write(keyboard_state_bytes)
        stream.close()
    """

    def __init__(self, conn: socket.socket, bus_id: int, dev_id: str) -> None:
        self._conn = conn
        self.bus_id = bus_id
        self.dev_id = dev_id
        self._closed = False

    @classmethod
    def open(
        cls,
        addr: str,
        bus_id: int,
        dev_id: str,
        timeout_s: float = 5.0,
    ) -> DeviceStream:
        """Connect to an existing device's stream.

        The device must already exist on the bus (use ``device_add`` first).
        Must be called within the connect timeout (default 5s) after adding.

        Args:
            addr: VIIPER server address (``host:port``).
            bus_id: Bus ID the device is on.
            dev_id: Device ID string (e.g. ``"1"``).
            timeout_s: Connection timeout.

        Returns:
            An open DeviceStream ready for writing.
        """
        host, port_str = addr.split(":")
        s = socket.create_connection((host, int(port_str)), timeout=timeout_s)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Send stream handshake
        handshake = f"bus/{bus_id}/{dev_id}\x00".encode("utf-8")
        s.sendall(handshake)

        # Set a send timeout so write() never blocks indefinitely.
        # The existing try/except in attack_loop._attack_one catches
        # socket.timeout and logs [ATTACK] input error, allowing the
        # loop to continue instead of hanging for 10+ seconds.
        s.settimeout(1.0)

        return cls(s, bus_id, dev_id)

    def write(self, data: bytes) -> int:
        """Write binary data to the device stream.

        Args:
            data: Binary input report (device-specific wire format).

        Returns:
            Number of bytes sent.
        """
        if self._closed:
            raise RuntimeError("stream closed")
        return self._conn.sendall(data) or len(data)

    def read(self, bufsize: int = 1024) -> bytes:
        """Read binary feedback from the device stream.

        Blocks until data is available or the connection is closed.
        Returns empty bytes when the stream is closed by the server.

        Args:
            bufsize: Maximum number of bytes to read.

        Returns:
            Binary feedback data (device-specific format, e.g. LED state
            for keyboards, rumble state for controllers).
        """
        if self._closed:
            raise RuntimeError("stream closed")
        return self._conn.recv(bufsize)

    def close(self) -> None:
        """Close the stream connection."""
        if self._closed:
            return
        self._closed = True
        try:
            self._conn.close()
        except OSError:
            pass

    def __enter__(self) -> DeviceStream:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
