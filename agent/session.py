"""Per-thread session state for the LangGraph agent.

LangGraph's checkpointer keeps chat/tool message history per thread_id.
This module keeps the *extra* run context the graph needs:

  - which CSV is attached for this thread
  - default max_rows / skip_rows for score/compare
  - last known scoring_meta snapshot (path identity)
  - a title + timestamps so the UI can list past chats

Persisted to data/sessions.json so CSV hints survive API restarts
(alongside the SQLite LangGraph checkpointer).
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from config import SESSIONS_PATH

# Sidebar labels stay on one line.
MAX_TITLE_CHARS = 60


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ThreadSession:
    """Analyst session bound to one chat thread."""

    thread_id: str
    csv_path: Optional[str] = None
    max_rows: int = 3000
    skip_rows: int = 0
    # Last scoring_meta.json contents we noticed (optional diagnostics).
    scoring_meta: Optional[dict[str, Any]] = field(default=None)
    # Sidebar label (first question of the thread) + ordering timestamps.
    title: Optional[str] = None
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)


class SessionStore:
    """Thread-safe map of thread_id → ThreadSession, backed by JSON on disk."""

    def __init__(self, path=SESSIONS_PATH) -> None:
        self._lock = threading.Lock()
        self._path = path
        self._sessions: dict[str, ThreadSession] = {}
        self._load()

    def new_thread_id(self) -> str:
        return uuid.uuid4().hex

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(raw, dict):
            return
        for tid, payload in raw.items():
            if not isinstance(payload, dict):
                continue
            created = str(payload.get("created_at") or _utc_now())
            self._sessions[str(tid)] = ThreadSession(
                thread_id=str(tid),
                csv_path=payload.get("csv_path"),
                max_rows=int(payload.get("max_rows", 3000)),
                skip_rows=int(payload.get("skip_rows", 0)),
                scoring_meta=payload.get("scoring_meta"),
                title=payload.get("title"),
                created_at=created,
                updated_at=str(payload.get("updated_at") or created),
            )

    def _save_unlocked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            tid: asdict(session) for tid, session in self._sessions.items()
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def get(self, thread_id: str) -> ThreadSession:
        with self._lock:
            session = self._sessions.get(thread_id)
            if session is None:
                session = ThreadSession(thread_id=thread_id)
                self._sessions[thread_id] = session
                self._save_unlocked()
            return session

    def upsert(
        self,
        thread_id: str,
        *,
        csv_path: Optional[str] = None,
        max_rows: Optional[int] = None,
        skip_rows: Optional[int] = None,
        scoring_meta: Optional[dict[str, Any]] = None,
        title: Optional[str] = None,
        clear_csv: bool = False,
    ) -> ThreadSession:
        """Update fields for a thread; create the session if missing."""
        with self._lock:
            session = self._sessions.get(thread_id)
            if session is None:
                session = ThreadSession(thread_id=thread_id)
                self._sessions[thread_id] = session

            if clear_csv:
                session.csv_path = None
            elif csv_path is not None:
                session.csv_path = csv_path

            if max_rows is not None:
                session.max_rows = int(max_rows)
            if skip_rows is not None:
                session.skip_rows = int(skip_rows)
            if scoring_meta is not None:
                session.scoring_meta = scoring_meta
            # First question wins, so the sidebar label stays stable.
            if title and not session.title:
                session.title = title.strip()[:MAX_TITLE_CHARS]

            session.updated_at = _utc_now()
            self._save_unlocked()
            return session

    def list_sessions(self) -> list[ThreadSession]:
        """Most recently used first (sidebar order)."""
        with self._lock:
            return sorted(
                self._sessions.values(),
                key=lambda s: s.updated_at,
                reverse=True,
            )

    def reset(self, thread_id: str) -> None:
        """Drop session extras (New chat). Checkpointer history is separate."""
        with self._lock:
            self._sessions.pop(thread_id, None)
            self._save_unlocked()


# Process-wide store used by the API / graph runner.
SESSION_STORE = SessionStore()
