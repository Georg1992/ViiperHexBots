"""Low-level VIIPER TCP client.

Implements the null-byte-terminated request/response protocol for
managing virtual USB buses and devices.

Protocol:
    Request:  ``<path> [<payload>]\\x00``
    Response: Single JSON line (or empty), connection closed by server.
"""

from __future__ import annotations

import json
import socket
from typing import Any


class ViiperError(Exception):
    """VIIPER API error response (RFC 7807 Problem Details)."""

    def __init__(self, status: int, title: str, detail: str = "") -> None:
        self.status = status
        self.title = title
        self.detail = detail
        super().__init__(f"{status} {title}: {detail}")


class ViiperClient:
    """Manages a TCP connection to the VIIPER server for bus/device control.

    Each management request opens a new short-lived TCP connection.
    For device streams, use DeviceStream directly.
    """

    def __init__(self, addr: str = "127.0.0.1:3242", timeout_s: float = 5.0) -> None:
        self._host, self._port_str = addr.rsplit(":", 1)
        self._port = int(self._port_str)
        self._timeout_s = timeout_s

    # ── Ping ──────────────────────────────────────────────────────────

    def ping(self) -> dict[str, str]:
        """Check server identity and version.

        Returns ``{"server": "VIIPER", "version": "..."}``.
        """
        return self._do("ping")

    # ── Bus management ────────────────────────────────────────────────

    def bus_list(self) -> list[int]:
        """List all active virtual bus IDs."""
        resp = self._do("bus/list")
        return resp.get("buses", [])

    def bus_create(self, bus_id: int | None = None) -> int:
        """Create a new virtual USB bus.

        Args:
            bus_id: Optional specific bus ID. If None, server picks a free one.

        Returns:
            The created bus ID.
        """
        payload = str(bus_id) if bus_id is not None else None
        resp = self._do("bus/create", payload)
        return int(resp["busId"])

    def bus_remove(self, bus_id: int) -> int:
        """Remove a bus and all devices on it.

        Returns:
            The removed bus ID.
        """
        resp = self._do("bus/remove", str(bus_id))
        return int(resp["busId"])

    # ── Device management ─────────────────────────────────────────────

    def device_add(
        self,
        bus_id: int,
        device_type: str,
        *,
        id_vendor: int | None = None,
        id_product: int | None = None,
    ) -> dict[str, Any]:
        """Add a device to a bus.

        Args:
            bus_id: Bus to attach to.
            device_type: Device type string (e.g. ``"keyboard"``, ``"mouse"``).
            id_vendor: Optional USB vendor ID.
            id_product: Optional USB product ID.

        Returns:
            Device info dict with keys: ``busId``, ``devId``, ``vid``, ``pid``,
            ``type``, ``deviceSpecific``.
        """
        body: dict[str, Any] = {"type": device_type}
        if id_vendor is not None:
            body["idVendor"] = id_vendor
        if id_product is not None:
            body["idProduct"] = id_product
        resp = self._do(f"bus/{bus_id}/add", json.dumps(body))
        return resp

    def device_remove(self, bus_id: int, dev_id: str) -> dict[str, Any]:
        """Remove a device from a bus.

        Returns:
            Dict with ``busId`` and ``devId``.
        """
        return self._do(f"bus/{bus_id}/remove", dev_id)

    def devices_list(self, bus_id: int) -> list[dict[str, Any]]:
        """List all devices on a bus."""
        resp = self._do(f"bus/{bus_id}/list")
        return resp.get("devices", [])

    # ── Low-level request ─────────────────────────────────────────────

    def _do(self, path: str, payload: str | None = None) -> Any:
        """Send a null-terminated request and parse the JSON response."""
        # Build request bytes
        if payload:
            line = f"{path} {payload}\x00".encode("utf-8")
        else:
            line = f"{path}\x00".encode("utf-8")

        # Open short-lived TCP connection
        s = socket.create_connection(
            (self._host, self._port), timeout=self._timeout_s
        )
        try:
            s.sendall(line)
            resp_bytes = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                resp_bytes += chunk
        finally:
            s.close()

        text = resp_bytes.decode("utf-8").rstrip("\n")

        if not text:
            return {}

        # Check for RFC 7807 error response
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return {"_raw": text}

        if isinstance(obj, dict):
            status = obj.get("status", 0)
            title = obj.get("title", "")
            if status or title:
                raise ViiperError(status, title, obj.get("detail", ""))
        return obj
