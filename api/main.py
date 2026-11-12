#!/usr/bin/env python3
"""FastAPI entrypoint for the ESA Telemanom agent.

Routes:
  GET  /health
  GET  /samples              → prepared demo CSV list (friendly names)
  POST /agent/run            → JSON body (path-based; still used by CLI-style clients)
  POST /agent/run-upload     → multipart: question + optional uploaded CSV or sample_id

Upload flow (what the React UI uses):
  1. Analyst picks a prepared sample OR uploads their own CSV
  2. Browser sends multipart form to /agent/run-upload
  3. We save uploads under uploads/ (or use samples/ path)
  4. We call run_agent with skip_rows=0 on that whole small file

How to run:

  conda activate esa-agent
  cd ~/ESA_ADB/esa-telemanom-agent
  uvicorn api.main:app --reload --host 127.0.0.1 --port 8100
"""
from __future__ import annotations

import sys
import traceback
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Import path setup (same pattern as before)
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
from llm import run_agent  # noqa: E402

from api.samples_catalog import list_available_samples, resolve_sample_path  # noqa: E402
from api.schemas import AgentRunRequest, AgentRunResponse, ToolStep  # noqa: E402

app = FastAPI(
    title="ESA Telemanom Agent API",
    description="HTTP wrapper around the ESA Telemanom single agent.",
    version="0.2.0",
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
    allow_methods=["GET", "POST", "OPTIONS"],
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
    )


def _call_agent(
    question: str,
    csv_path: Optional[str],
    max_rows: int,
    skip_rows: int,
    k: int = 5,
) -> AgentRunResponse:
    """Run the agent and map errors to HTTPException."""
    try:
        result = run_agent(
            question,
            k=k,
            model=None,
            csv_hint=csv_path,
            default_max_rows=max_rows,
            default_skip_rows=skip_rows,
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


def _save_upload(upload: UploadFile) -> Path:
    """Save an uploaded CSV into uploads/ and return the absolute path."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    original_name = upload.filename or "telemetry.csv"
    # Keep only the file name (no directories from the client).
    safe_name = Path(original_name).name
    if not safe_name.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV telemetry file.",
        )

    # Unique prefix so two users do not overwrite each other.
    dest = UPLOADS_DIR / f"{uuid.uuid4().hex}_{safe_name}"

    # Read and write in a simple loop (clearer than fancy streaming helpers).
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    dest.write_bytes(data)
    return dest


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Simple liveness check for developers / scripts."""
    return {"status": "ok", "service": "esa-telemanom-agent-api"}


@app.get("/samples")
def samples() -> dict:
    """List prepared sample files with analyst-friendly titles.

    The UI uses this to offer "try a prepared window" without showing paths.
    """
    return {"samples": list_available_samples()}


@app.post("/agent/run", response_model=AgentRunResponse)
def agent_run(body: AgentRunRequest) -> AgentRunResponse:
    """JSON entrypoint (path-based). Kept for curl / advanced clients."""
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    return _call_agent(
        question=question,
        csv_path=body.csv_path,
        max_rows=body.max_rows,
        skip_rows=body.skip_rows,
        k=body.k,
    )


@app.post("/agent/run-upload", response_model=AgentRunResponse)
async def agent_run_upload(
    question: str = Form(..., description="User question"),
    # Optional: browser file upload
    file: Optional[UploadFile] = File(None),
    # Optional: id of a prepared sample under samples/
    sample_id: Optional[str] = Form(None),
) -> AgentRunResponse:
    """Main UI entrypoint: question + optional telemetry CSV.

    Rules:
      - If the user uploads a file, we save it and score that whole file.
      - Else if sample_id is set, we use that prepared sample file.
      - Else RAG-only (no CSV) — good for "ask about the model" questions.
      - Prepared samples are already small, so skip_rows is always 0.
    """
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    csv_path: Optional[str] = None

    # Prefer an explicit upload over a sample id if both are sent.
    if file is not None and file.filename:
        saved = _save_upload(file)
        csv_path = str(saved)
    elif sample_id:
        sample_id = sample_id.strip()
        if sample_id:
            try:
                csv_path = str(resolve_sample_path(sample_id))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Whole small file: no skipping. max_rows high enough for our samples.
    # (Uploaded analyst files should also be modest slices, not multi-GB.)
    return _call_agent(
        question=question,
        csv_path=csv_path,
        max_rows=20000,
        skip_rows=0,
        k=5,
    )
