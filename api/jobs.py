"""Background agent jobs so long Telemanom runs do not block the HTTP request.

Flow:
  1. POST /agent/jobs  → queued job_id (returns immediately)
  2. Worker thread runs run_agent(...) and pushes live progress messages
  3. GET /agent/jobs/{id} → status + progress_steps + result when done

Finished jobs are pruned by TTL / max count so RAM does not grow forever.
In-process only (fine for single uvicorn worker).
"""
from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from config import JOB_MAX_KEPT, JOB_TTL_SECONDS

# One worker: TensorFlow scoring is heavy; avoid parallel model loads.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="esa-agent-job")
_LOCK = threading.Lock()
_JOBS: dict[str, "AgentJob"] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return datetime.now(timezone.utc)


@dataclass
class AgentJob:
    job_id: str
    thread_id: str
    question: str
    status: str = "queued"  # queued | running | done | error
    status_message: str = "Queued"
    progress_steps: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    def touch(self, status: str, message: str) -> None:
        self.status = status
        self.status_message = message
        self.updated_at = _utc_now()
        # Keep a short trail for the UI checklist (skip exact duplicates).
        if not self.progress_steps or self.progress_steps[-1] != message:
            self.progress_steps.append(message)
            if len(self.progress_steps) > 40:
                self.progress_steps = self.progress_steps[-40:]


def _purge_jobs_unlocked() -> None:
    """Drop old finished jobs; keep active ones. Call while holding _LOCK."""
    now = datetime.now(timezone.utc)
    expired: list[str] = []
    for jid, job in _JOBS.items():
        if job.status not in ("done", "error"):
            continue
        age = (now - _parse_iso(job.updated_at)).total_seconds()
        if age > JOB_TTL_SECONDS:
            expired.append(jid)
    for jid in expired:
        _JOBS.pop(jid, None)

    # Hard cap: remove oldest finished jobs first.
    if len(_JOBS) <= JOB_MAX_KEPT:
        return
    finished = sorted(
        (
            (jid, job)
            for jid, job in _JOBS.items()
            if job.status in ("done", "error")
        ),
        key=lambda item: item[1].updated_at,
    )
    overflow = len(_JOBS) - JOB_MAX_KEPT
    for jid, _ in finished[:overflow]:
        _JOBS.pop(jid, None)


def get_job(job_id: str) -> Optional[AgentJob]:
    with _LOCK:
        _purge_jobs_unlocked()
        return _JOBS.get(job_id)


def submit_agent_job(
    *,
    question: str,
    thread_id: str,
    csv_path: Optional[str],
    max_rows: int,
    skip_rows: int,
    k: int = 5,
) -> AgentJob:
    """Enqueue an agent run; returns immediately with status=queued."""
    job_id = uuid.uuid4().hex
    job = AgentJob(
        job_id=job_id,
        thread_id=thread_id,
        question=question,
        status="queued",
        status_message="Queued — waiting for a free worker",
        progress_steps=["Queued — waiting for a free worker"],
    )
    with _LOCK:
        _purge_jobs_unlocked()
        _JOBS[job_id] = job

    _EXECUTOR.submit(
        _run_job,
        job_id,
        question,
        thread_id,
        csv_path,
        max_rows,
        skip_rows,
        k,
    )
    return job


def _run_job(
    job_id: str,
    question: str,
    thread_id: str,
    csv_path: Optional[str],
    max_rows: int,
    skip_rows: int,
    k: int,
) -> None:
    job = get_job(job_id)
    if job is None:
        return

    job.touch("running", "Starting assistant…")

    from progress import reset_progress_reporter, set_progress_reporter

    def _on_progress(message: str) -> None:
        if job.status in ("done", "error"):
            return
        job.touch("running", message)

    token = set_progress_reporter(_on_progress)
    try:
        from llm import run_agent

        result = run_agent(
            question,
            k=k,
            model=None,
            csv_hint=csv_path,
            default_max_rows=max_rows,
            default_skip_rows=skip_rows,
            thread_id=thread_id,
        )
        job.result = result
        job.touch("done", "Done")
    except SystemExit as exc:
        job.error = str(exc) if str(exc) else "Assistant is not configured (missing API key)."
        job.touch("error", "Configuration error")
    except Exception as exc:
        print(f"ERROR in agent job {job_id}:")
        traceback.print_exc()
        job.error = f"{type(exc).__name__}: {exc}"
        job.touch("error", "Assistant failed")
    finally:
        reset_progress_reporter(token)
