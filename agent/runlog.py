"""One JSON line per agent turn, for "why was this slow / wrong" questions.

Append-only `data/runs.jsonl`, rotated once at RUNLOG_MAX_BYTES. Logging must
never break a run, so every failure here is swallowed.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from config import RUNLOG_MAX_BYTES, RUNLOG_PATH

_LOCK = threading.Lock()

# Questions can carry operational detail; keep enough to identify the turn.
MAX_QUESTION_CHARS = 300


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rotate_unlocked(path: Path) -> None:
    try:
        if path.is_file() and path.stat().st_size >= RUNLOG_MAX_BYTES:
            path.replace(path.with_suffix(path.suffix + ".1"))
    except OSError:
        pass


def log_turn(
    *,
    thread_id: str,
    question: str,
    route: Optional[str] = None,
    tools: Optional[Sequence[str]] = None,
    duration_ms: Optional[int] = None,
    provider: str = "",
    model: str = "",
    max_rows: Optional[int] = None,
    reused_alarms: Optional[bool] = None,
    unverified: Optional[Sequence[str]] = None,
    answer_chars: Optional[int] = None,
    error: Optional[str] = None,
    path: Optional[Path] = None,
) -> None:
    """Append one turn record. Never raises."""
    record: dict[str, Any] = {
        "ts": _utc_now(),
        "thread_id": thread_id,
        "question": (question or "")[:MAX_QUESTION_CHARS],
        "route": route,
        "tools": list(tools or []),
        "duration_ms": duration_ms,
        "provider": provider,
        "model": model,
        "max_rows": max_rows,
        "reused_alarms": reused_alarms,
        "unverified": list(unverified or []),
        "unverified_count": len(list(unverified or [])),
        "answer_chars": answer_chars,
        "error": error,
    }

    target = path or RUNLOG_PATH
    try:
        with _LOCK:
            target.parent.mkdir(parents=True, exist_ok=True)
            _rotate_unlocked(target)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def read_turns(limit: int = 50, path: Optional[Path] = None) -> list[dict]:
    """Most recent turns first (diagnostics / tests). Never raises."""
    target = path or RUNLOG_PATH
    try:
        if not target.is_file():
            return []
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    out: list[dict] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(out) >= limit:
            break
    return out
