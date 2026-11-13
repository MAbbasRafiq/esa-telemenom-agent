"""Request and response shapes for the FastAPI routes.

Why a separate file?
  - main.py stays focused on "routes + wiring"
  - These models document the JSON contract for the future React UI
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    """JSON body the client sends to POST /agent/run.

    Same ideas as the CLI flags on agent/agent.py:
      question          → positional CLI argument
      k                 → -k
      model             → --model
      csv_path          → --csv
      max_rows          → --max-rows
      skip_rows         → --skip-rows
    """

    # The user question (required). Example: "What is the window size?"
    question: str = Field(
        ...,
        description="Natural language question for the agent",
        min_length=1,
    )

    # How many RAG chunks to retrieve by default (tool may still pass its own k).
    k: int = Field(
        default=5,
        ge=1,
        le=8,
        description="Default RAG top-k (capped at 8, same as the CLI agent)",
    )

    # Optional override for the LLM model name (usually leave empty).
    model: Optional[str] = Field(
        default=None,
        description="Optional LLM model override (e.g. a Groq model id)",
    )

    # Optional path to a telemetry CSV. Used as the default for score/compare tools.
    csv_path: Optional[str] = Field(
        default=None,
        description="Default telemetry CSV path for score_telemetry / compare tools",
    )

    # How many rows to read from the CSV when scoring (keeps runs fast).
    max_rows: int = Field(
        default=3000,
        ge=1,
        description="Max rows to score from the CSV slice",
    )

    # How many rows to skip from the start of the CSV before scoring.
    skip_rows: int = Field(
        default=0,
        ge=0,
        description="Rows to skip from the start of the CSV",
    )

    # LangGraph thread id — same id keeps chat/tool memory across asks.
    thread_id: Optional[str] = Field(
        default=None,
        description="Optional session thread id (returned by the API; reuse for follow-ups)",
    )


class ToolStep(BaseModel):
    """One tool call the agent made during the ReAct loop."""

    # Tool name, e.g. "retrieve_knowledge"
    tool: str

    # Arguments the LLM passed to that tool (JSON object)
    args: dict[str, Any]

    # Raw string the tool returned (may be long for score/compare)
    observation: str


class AgentRunResponse(BaseModel):
    """JSON the API returns after the agent finishes.

    Matches the dict from llm.run_agent():
      provider, model, tool_steps, final_answer, thread_id
    """

    provider: str
    model: str
    tool_steps: list[ToolStep]
    final_answer: str
    thread_id: Optional[str] = Field(
        default=None,
        description="Session thread id — send this back on the next ask",
    )
    routed_to: Optional[str] = Field(
        default=None,
        description="Multi-agent route: knowledge | score | compare",
    )
    unverified_claims: list[str] = Field(
        default_factory=list,
        description="Numbers in the answer that no tool observation supports",
    )


class SessionInfo(BaseModel):
    """One past chat, for the sidebar list."""

    thread_id: str
    title: Optional[str] = None
    csv_path: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SessionTurn(BaseModel):
    """A question and its answer, replayed from checkpointed history."""

    question: str
    answer: str = ""
    tools: list[str] = Field(default_factory=list)


class SessionDetail(SessionInfo):
    """A past chat with its messages, so the UI can reopen it."""

    turns: list[SessionTurn] = Field(default_factory=list)


class JobAccepted(BaseModel):
    """Returned immediately when an async agent job is queued."""

    job_id: str
    thread_id: str
    status: str = "queued"
    status_message: str = "Queued"


class JobStatusResponse(BaseModel):
    """Poll response for GET /agent/jobs/{job_id}."""

    job_id: str
    thread_id: str
    status: str  # queued | running | done | error
    status_message: str
    progress_steps: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    result: Optional[AgentRunResponse] = None
    error: Optional[str] = None
