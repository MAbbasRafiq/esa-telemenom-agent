"""Live progress reporting for long agent / scoring runs.

Jobs set a callback via set_progress_reporter(); tools and the graph call
report_progress("…") so the UI can poll changing status_message / steps.
"""
from __future__ import annotations

import contextvars
from typing import Callable, Optional

ProgressFn = Callable[[str], None]

_REPORTER: contextvars.ContextVar[Optional[ProgressFn]] = contextvars.ContextVar(
    "esa_progress_reporter",
    default=None,
)


def set_progress_reporter(fn: Optional[ProgressFn]):
    """Bind a reporter for the current worker context; returns a reset token."""
    return _REPORTER.set(fn)


def reset_progress_reporter(token) -> None:
    _REPORTER.reset(token)


def report_progress(message: str) -> None:
    """Update the active job UI message (no-op if no reporter bound)."""
    fn = _REPORTER.get()
    if fn is None:
        return
    text = (message or "").strip()
    if not text:
        return
    try:
        fn(text)
    except Exception:
        # Never break scoring because of a UI progress glitch.
        pass
