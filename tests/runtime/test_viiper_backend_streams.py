"""ViiperBackend must keep streams open across hunt stop/start."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from pybot.runtime.input import viiper_backend as vb
from pybot.runtime.input.viiper_backend import ViiperBackend


class ViiperBackendStreamLifetimeTests(unittest.TestCase):
    def tearDown(self) -> None:
        ViiperBackend.close_shared_streams()

    def test_shutdown_releases_keys_without_closing_shared_streams(self) -> None:
        backend = ViiperBackend()
        kb = MagicMock()
        mouse = MagicMock()
        with vb._shared_lock:
            vb._shared_kb = kb
            vb._shared_mouse = mouse
        backend._kb_stream = kb
        backend._mouse_stream = mouse
        backend._connected = True

        backend.shutdown()

        kb.write.assert_called_once()
        kb.close.assert_not_called()
        mouse.close.assert_not_called()
        self.assertIs(vb._shared_kb, kb)
        self.assertIs(vb._shared_mouse, mouse)

    def test_connect_reuses_shared_streams(self) -> None:
        kb = MagicMock()
        mouse = MagicMock()
        with vb._shared_lock:
            vb._shared_kb = kb
            vb._shared_mouse = mouse

        first = ViiperBackend()
        first.connect()
        second = ViiperBackend()
        second.connect()

        self.assertIs(first._kb_stream, kb)
        self.assertIs(second._kb_stream, kb)
        self.assertTrue(first._connected)
        self.assertTrue(second._connected)

    def test_close_shared_streams_closes_tcp(self) -> None:
        kb = MagicMock()
        mouse = MagicMock()
        with vb._shared_lock:
            vb._shared_kb = kb
            vb._shared_mouse = mouse

        ViiperBackend.close_shared_streams()

        kb.close.assert_called_once()
        mouse.close.assert_called_once()
        self.assertIsNone(vb._shared_kb)
        self.assertIsNone(vb._shared_mouse)


if __name__ == "__main__":
    unittest.main()
