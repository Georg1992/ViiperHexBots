"""ViiperBackend must keep streams open across hunt stop/start."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock

from pybot.runtime.input.viiper_backend import (
    ViiperBackend,
    ViiperStreamStore,
)


class ViiperBackendStreamLifetimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = ViiperStreamStore()

    def tearDown(self) -> None:
        self.store.close()

    def _seed_shared_streams(self, kb, mouse) -> None:
        """Record already-open streams on the shared store (test seam)."""
        self.store.open_once(lambda: (kb, mouse))

    def test_shutdown_releases_keys_without_closing_shared_streams(self) -> None:
        backend = ViiperBackend(stream_store=self.store)
        kb = MagicMock()
        mouse = MagicMock()
        self._seed_shared_streams(kb, mouse)
        backend._kb_stream = kb
        backend._mouse_stream = mouse
        backend._connected = True

        backend.shutdown()

        kb.write.assert_called_once()
        kb.close.assert_not_called()
        mouse.close.assert_not_called()
        # The shared store must retain the streams so Stop/Start never
        # triggers VIIPER device auto-removal: a fresh backend reuses them.
        revived = ViiperBackend(stream_store=self.store)
        revived.connect()
        self.assertIs(revived._kb_stream, kb)
        self.assertIs(revived._mouse_stream, mouse)

    def test_connect_reuses_shared_streams(self) -> None:
        kb = MagicMock()
        mouse = MagicMock()
        self._seed_shared_streams(kb, mouse)

        first = ViiperBackend(stream_store=self.store)
        first.connect()
        second = ViiperBackend(stream_store=self.store)
        second.connect()

        self.assertIs(first._kb_stream, kb)
        self.assertIs(second._kb_stream, kb)
        self.assertTrue(first._connected)
        self.assertTrue(second._connected)

    def test_shutdown_releases_mouse_buttons_as_well_as_keys(self) -> None:
        backend = ViiperBackend(stream_store=self.store)
        kb = MagicMock()
        mouse = MagicMock()
        self._seed_shared_streams(kb, mouse)
        backend._kb_stream = kb
        backend._mouse_stream = mouse
        backend._connected = True
        backend._mouse_buttons = 1

        backend.shutdown()

        kb.write.assert_called_once()
        mouse.write.assert_called_once()
        self.assertEqual(backend._mouse_buttons, 0)

    def test_move_and_double_click_emits_two_atomic_clicks(self) -> None:
        backend = ViiperBackend(stream_store=self.store)
        backend._connected = True
        backend._kb_stream = MagicMock()
        backend._mouse_stream = MagicMock()
        backend._cancel_event.clear()

        with unittest.mock.patch(
            "pybot.runtime.input.viiper_backend.user32"
        ) as mock_user32, unittest.mock.patch.object(
            backend, "_wait_or_cancel", return_value=True
        ), unittest.mock.patch.object(
            backend, "_mouse_button"
        ) as mouse_button:
            self.assertTrue(backend.move_and_double_click(120, 240))

        mock_user32.SetCursorPos.assert_called_once_with(120, 240)
        self.assertEqual(
            [call for call in mouse_button.call_args_list],
            [
                unittest.mock.call(0, down=True),
                unittest.mock.call(0, down=False),
                unittest.mock.call(0, down=True),
                unittest.mock.call(0, down=False),
            ],
        )

    def test_cancel_interrupts_key_tap_and_releases_key(self) -> None:
        backend = ViiperBackend(stream_store=self.store)
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
        backend = ViiperBackend(stream_store=self.store)
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
        backend = ViiperBackend(stream_store=self.store)
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

    def test_store_close_closes_tcp_streams(self) -> None:
        kb = MagicMock()
        mouse = MagicMock()
        self._seed_shared_streams(kb, mouse)

        self.store.close()

        kb.close.assert_called_once()
        mouse.close.assert_called_once()

    def test_open_once_opens_streams_exactly_once(self) -> None:
        store = ViiperStreamStore()
        opened = 0

        def _opener():
            nonlocal opened
            opened += 1
            return MagicMock(), MagicMock()

        first_kb, first_mouse = store.open_once(_opener)
        second_kb, second_mouse = store.open_once(_opener)

        self.assertEqual(opened, 1)
        self.assertIs(first_kb, second_kb)
        self.assertIs(first_mouse, second_mouse)


if __name__ == "__main__":
    unittest.main()
