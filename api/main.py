#!/usr/bin/env python3
"""FastAPI entrypoint for the ESA Telemanom agent.

Routes:
  GET  /health
  GET  /samples
  GET  /sessions             → past chats (sidebar list)
  GET  /sessions/{thread_id} → one chat replayed from checkpoints
  DEL  /sessions/{thread_id} → forget a chat
  POST /agent/run            → sync JSON (CLI-style)
  POST /agent/run-upload     → sync multipart (compat)
  POST /agent/jobs           → async multipart (UI): returns job_id immediately
  GET  /agent/jobs/{job_id}  → poll status + result

How to run:

  conda activate esa-agent
  cd ~/ESA_ADB/esa-telemanom-agent
  uvicorn api.main:app --reload --host 127.0.0.1 --port 8100
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Import path setup
# ---------------------------------------------------------------------------

API_DIR = Path(__file__).resolve().parent
ROOT = API_DIR.parent
AGENT_DIR = ROOT / "agent"
UPLOADS_DIR = ROOT / "uploads"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import config  # noqa: E402, F401
from config import MAX_UI_SCORE_ROWS  # noqa: E402
from llm import run_agent  # noqa: E402
from session import SESSION_STORE  # noqa: E402

from api.csv_guards import (  # noqa: E402
    clamp_score_rows,
    save_upload_checked,
    validate_existing_csv_rows,
)
from api.jobs import get_job, submit_agent_job  # noqa: E402
from api.samples_catalog import list_available_samples, resolve_sample_path  # noqa: E402
from api.schemas import (  # noqa: E402
    AgentRunRequest,
    AgentRunResponse,
    JobAccepted,
    JobStatusResponse,
    SessionDetail,
    SessionInfo,
    SessionTurn,
    ToolStep,
)

app = FastAPI(
    title="ESA Telemanom Agent API",
    description=(
        "LangGraph multi-agent (Router → Knowledge/Score/Compare) with "
        "durable sessions (SQLite) and async jobs."
    ),
    version="0.5.0",
)

ALLOWED_ORIGINS = [
    "http://localhost:5180",
    "http://127.0.0.1:5180",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_to_response(result: dict) -> AgentRunResponse:
    """Convert the plain dict from run_agent() into the API response model."""
    tool_steps: list[ToolStep] = []
    for step in result.get("tool_steps", []):
        tool_steps.append(
            ToolStep(
                tool=step.get("tool", ""),
                args=step.get("args", {}),
                observation=step.get("observation", ""),
            )
        )
    return AgentRunResponse(
        provider=result.get("provider", ""),
        model=result.get("model", ""),
        tool_steps=tool_steps,
        final_answer=result.get("final_answer", ""),
        thread_id=result.get("thread_id"),
        routed_to=result.get("routed_to"),
        unverified_claims=list(result.get("unverified_claims") or []),
    )


def _resolve_csv_and_thread(
    *,
    file: Optional[UploadFile],
    sample_id: Optional[str],
    thread_id: Optional[str],
) -> tuple[Optional[str], str]:
    """Pick CSV path + ensure a thread_id (reuse prior CSV on follow-ups)."""
    tid = (thread_id or "").strip() or SESSION_STORE.new_thread_id()
    csv_path: Optional[str] = None

    if file is not None and file.filename:
        saved = save_upload_checked(file, UPLOADS_DIR)
        csv_path = str(saved)
    elif sample_id:
        sample_id = sample_id.strip()
        if sample_id:
            try:
                csv_path = str(resolve_sample_path(sample_id))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        prior = SESSION_STORE.get(tid)
        csv_path = prior.csv_path

    validate_existing_csv_rows(csv_path)

    SESSION_STORE.upsert(
        tid,
        csv_path=csv_path,
        max_rows=MAX_UI_SCORE_ROWS,
        skip_rows=0,
    )
    return csv_path, tid


def _call_agent(
    question: str,
    csv_path: Optional[str],
    max_rows: int,
    skip_rows: int,
    k: int = 5,
    thread_id: Optional[str] = None,
) -> AgentRunResponse:
    """Run the agent synchronously and map errors to HTTPException."""
    tid = (thread_id or "").strip() or SESSION_STORE.new_thread_id()
    max_rows = clamp_score_rows(max_rows, ui=False) or max_rows
    SESSION_STORE.upsert(
        tid,
        csv_path=csv_path,
        max_rows=max_rows,
        skip_rows=skip_rows,
    )
    try:
        result = run_agent(
            question,
            k=k,
            model=None,
            csv_hint=csv_path,
            default_max_rows=max_rows,
            default_skip_rows=skip_rows,
            thread_id=tid,
        )
        return _result_to_response(result)
    except SystemExit as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc) if str(exc) else "Assistant is not configured (missing API key).",
        ) from exc
    except Exception as exc:
        print("ERROR while running agent:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Assistant failed: {type(exc).__name__}: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Simple liveness check for developers / scripts."""
    return {"status": "ok", "service": "esa-telemanom-agent-api"}


@app.get("/samples")
def samples() -> dict:
    """List prepared sample files with analyst-friendly titles."""
    return {"samples": list_available_samples()}


def _session_info(session) -> SessionInfo:
    return SessionInfo(
        thread_id=session.thread_id,
        title=session.title,
        csv_path=session.csv_path,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@app.get("/sessions")
def sessions_list() -> dict:
    """Past chats, newest first. Untitled threads (no question yet) are hidden."""
    items = [
        _session_info(session)
        for session in SESSION_STORE.list_sessions()
        if session.title
    ]
    return {"sessions": items}


@app.get("/sessions/{thread_id}", response_model=SessionDetail)
def session_detail(thread_id: str) -> SessionDetail:
    """Reopen a past chat: session info + replayed question/answer turns."""
    from graph import load_thread_turns

    sessions = {s.thread_id: s for s in SESSION_STORE.list_sessions()}
    session = sessions.get(thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown chat.")

    turns = [
        SessionTurn(
            question=turn.get("question", ""),
            answer=turn.get("answer", ""),
            tools=list(turn.get("tools") or []),
        )
        for turn in load_thread_turns(thread_id)
    ]
    base = _session_info(session)
    return SessionDetail(**base.model_dump(), turns=turns)


@app.delete("/sessions/{thread_id}")
def session_delete(thread_id: str) -> dict:
    """Forget a chat (session extras + checkpointed history)."""
    from graph import delete_thread

    delete_thread(thread_id)
    return {"status": "deleted", "thread_id": thread_id}


@app.post("/agent/run", response_model=AgentRunResponse)
def agent_run(body: AgentRunRequest) -> AgentRunResponse:
    """JSON entrypoint (path-based, sync). Kept for curl / advanced clients."""
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    return _call_agent(
        question=question,
        csv_path=body.csv_path,
        max_rows=body.max_rows,
        skip_rows=body.skip_rows,
        k=body.k,
        thread_id=body.thread_id,
    )


@app.post("/agent/run-upload", response_model=AgentRunResponse)
async def agent_run_upload(
    question: str = Form(..., description="User question"),
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
) -> AgentRunResponse:
    """Sync multipart entrypoint (compat). Prefer POST /agent/jobs for the UI."""
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    csv_path, tid = _resolve_csv_and_thread(
        file=file,
        sample_id=sample_id,
        thread_id=thread_id,
    )
    return _call_agent(
        question=question,
        csv_path=csv_path,
        max_rows=MAX_UI_SCORE_ROWS,
        skip_rows=0,
        k=5,
        thread_id=tid,
    )


@app.post("/agent/jobs", response_model=JobAccepted)
async def agent_job_start(
    question: str = Form(..., description="User question"),
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
) -> JobAccepted:
    """Queue an agent run and return immediately (UI uses this + poll)."""
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    csv_path, tid = _resolve_csv_and_thread(
        file=file,
        sample_id=sample_id,
        thread_id=thread_id,
    )
    job = submit_agent_job(
        question=question,
        thread_id=tid,
        csv_path=csv_path,
        max_rows=MAX_UI_SCORE_ROWS,
        skip_rows=0,
        k=5,
    )
    return JobAccepted(
        job_id=job.job_id,
        thread_id=job.thread_id,
        status=job.status,
        status_message=job.status_message,
    )


@app.get("/agent/jobs/{job_id}", response_model=JobStatusResponse)
def agent_job_status(job_id: str) -> JobStatusResponse:
    """Poll a background agent job until status is done or error."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")

    result: Optional[AgentRunResponse] = None
    if job.status == "done" and job.result is not None:
        result = _result_to_response(job.result)

    return JobStatusResponse(
        job_id=job.job_id,
        thread_id=job.thread_id,
        status=job.status,
        status_message=job.status_message,
        progress_steps=list(job.progress_steps),
        created_at=job.created_at,
        updated_at=job.updated_at,
        result=result,
        error=job.error,
    )
