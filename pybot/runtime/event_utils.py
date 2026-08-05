"""Helpers for reading runtime event state defensively.

Runtime workers and actions tolerate lightweight test contexts (MagicMocks)
where a real :class:`threading.Event` is expected. A mock's ``is_set()``
returns a truthy mock rather than a boolean, so every direct read would
otherwise repeat the same ``getattr`` / ``type() is bool`` dance.
``event_is_set`` centralises that policy in one place: real events report
their boolean state, anything else reports ``None``.
"""

from __future__ import annotations


def event_is_set(event) -> bool | None:
    """Return an event's boolean state, or ``None`` when no real event exists.

    ``None`` means the value is not a real event (a mock, ``None``, or an
    object without ``is_set``). Callers that need a strict boolean should
    compare against ``is True`` / ``is False``.
    """
    is_set = getattr(event, "is_set", None)
    if not callable(is_set):
        return None
    try:
        value = is_set()
    except Exception:
        return None
    return value if type(value) is bool else None
