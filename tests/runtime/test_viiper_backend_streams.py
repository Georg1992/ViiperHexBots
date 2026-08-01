"""ViiperBackend must keep streams open across hunt stop/start."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock

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

    def test_shutdown_releases_mouse_buttons_as_well_as_keys(self) -> None:
        backend = ViiperBackend()
        kb = MagicMock()
        mouse = MagicMock()
        with vb._shared_lock:
            vb._shared_kb = kb
            vb._shared_mouse = mouse
        backend._kb_stream = kb
        backend._mouse_stream = mouse
        backend._connected = True
        backend._mouse_buttons = 1

        backend.shutdown()

        kb.write.assert_called_once()
        mouse.write.assert_called_once()
        self.assertEqual(backend._mouse_buttons, 0)

    def test_cancel_interrupts_key_tap_and_releases_key(self) -> None:
        backend = ViiperBackend()
        backend._connected = True
        backend._kb_stream = MagicMock()
        backend._mouse_stream = MagicMock()
        backend._cancel_event.clear()
        result: list[bool] = []

        thread = threading.Thread(
            target=lambda: result.append(
                backend.key_tap(82, press_s=1.0, after_s=5.0)
            ),
            daemon=True,
        )
        thread.start()
        time.sleep(0.05)
        backend.cancel_pending()
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [False])
        self.assertEqual(
            backend._kb_stream.write.call_count,
            2,
        )

    def test_toggle_key_stays_accepted_after_cancellation(self) -> None:
        backend = ViiperBackend()
        backend._connected = True
        backend._kb_stream = MagicMock()
        backend._mouse_stream = MagicMock()
        backend._cancel_event.clear()
        result: list[bool] = []

        thread = threading.Thread(
            target=lambda: result.append(backend.toggle_key(82)),
            daemon=True,
        )
        thread.start()
        time.sleep(0.01)
        backend.cancel_pending()
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [True])
        self.assertEqual(backend._kb_stream.write.call_count, 2)

    def test_cancelled_session_still_releases_left_button(self) -> None:
        backend = ViiperBackend()
        kb = MagicMock()
        mouse = MagicMock()
        backend._kb_stream = kb
        backend._mouse_stream = mouse
        backend._connected = True
        backend._cancel_event.set()
        backend._mouse_buttons = 1

        self.assertTrue(backend.set_left_button(False))
        self.assertEqual(backend._mouse_buttons, 0)
        mouse.write.assert_called_once()

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
