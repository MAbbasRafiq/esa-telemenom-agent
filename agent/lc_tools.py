"""LangChain tool wrappers around the existing tools.py functions.

Contracts stay in tools.py / docs/TOOL_CONTRACTS.md — this file only adapts
them for LangGraph (StructuredTool) and trims long observations for TPM limits.
"""
from __future__ import annotations

from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from tools import (
    tool_compare_alarms_to_labels,
    tool_retrieve_knowledge,
    tool_score_telemetry,
)

# Cap tool text fed back into the LLM (Groq free/dev TPM is tight).
MAX_OBSERVATION_CHARS = 3500


def trim_observation(text: str, limit: int = MAX_OBSERVATION_CHARS) -> str:
    """Shorten long score/compare dumps before they hit the LLM context."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…[truncated for context limit]"


def resolve_slice(
    csv_path: str,
    max_rows: Optional[int],
    skip_rows: Optional[int],
) -> tuple[str, Optional[int], int]:
    """Pin csv/slice to this thread's session values.

    Specialists (score vs compare) otherwise pass different defaults, which
    would look like a new slice and force a second Telemanom pass.
    """
    from multi_agent import ACTIVE_THREAD
    from session import SESSION_STORE

    thread_id = ACTIVE_THREAD.get()
    session = SESSION_STORE.get(thread_id) if thread_id else None
    if session is None:
        return csv_path, max_rows, int(skip_rows or 0)

    path = csv_path or (session.csv_path or "")
    # Session owns the slice so score → compare reuse works across turns.
    return path, session.max_rows, session.skip_rows


class RetrieveArgs(BaseModel):
    query: str = Field(..., description="Search query over lab knowledge")
    k: int = Field(default=5, description="Top-k chunks (1–8)")


class ScoreArgs(BaseModel):
    csv_path: str = Field(..., description="Path to telemetry CSV")
    max_rows: Optional[int] = Field(default=3000, description="Max rows to score")
    skip_rows: int = Field(default=0, description="Rows to skip from start")
    threshold: bool = Field(default=True, description="Apply alarm thresholds")


class CompareArgs(BaseModel):
    csv_path: str = Field(..., description="Path to telemetry CSV with labels")
    max_rows: Optional[int] = Field(default=3000, description="Max rows (same as score)")
    skip_rows: int = Field(default=0, description="Rows to skip (same as score)")
    re_score: bool = Field(
        default=False,
        description="Usually false; scores only when alarms missing or file changed",
    )


def _retrieve(query: str, k: int = 5) -> str:
    from progress import report_progress

    report_progress("Tool: retrieve_knowledge")
    out = trim_observation(tool_retrieve_knowledge(query, k=k))
    report_progress("Knowledge search done — reasoning…")
    return out


def _score(
    csv_path: str,
    max_rows: Optional[int] = 3000,
    skip_rows: int = 0,
    threshold: bool = True,
) -> str:
    from progress import report_progress

    report_progress("Tool: score_telemetry")
    path, rows, skip = resolve_slice(csv_path, max_rows, skip_rows)
    out = trim_observation(
        tool_score_telemetry(
            csv_path=path,
            max_rows=rows,
            skip_rows=skip,
            threshold=threshold,
        )
    )
    report_progress("Scoring tool done — reasoning…")
    return out


def _compare(
    csv_path: str,
    max_rows: Optional[int] = 3000,
    skip_rows: int = 0,
    re_score: bool = False,
) -> str:
    from progress import report_progress

    report_progress("Tool: compare_alarms_to_labels")
    path, rows, skip = resolve_slice(csv_path, max_rows, skip_rows)
    out = trim_observation(
        tool_compare_alarms_to_labels(
            csv_path=path,
            max_rows=rows,
            skip_rows=skip,
            re_score=re_score,
        )
    )
    report_progress("Compare done — writing answer…")
    return out


def build_langchain_tools() -> list[StructuredTool]:
    """Three tools matching TOOLS_SPEC / TOOL_CONTRACTS.md."""
    return [
        StructuredTool.from_function(
            func=_retrieve,
            name="retrieve_knowledge",
            description=(
                "Search lab knowledge (channels, anomaly types, model card, lab results). "
                "Keep k between 1 and 8."
            ),
            args_schema=RetrieveArgs,
        ),
        StructuredTool.from_function(
            func=_score,
            name="score_telemetry",
            description=(
                "Run Telemanom-ESA on a telemetry CSV. Returns which channels alarm "
                "(short summary). No label metrics / ESAScores. "
                "Reuses alarms if this CSV slice was already scored. "
                "Do not also call compare_alarms_to_labels in the same turn."
            ),
            args_schema=ScoreArgs,
        ),
        StructuredTool.from_function(
            func=_compare,
            name="compare_alarms_to_labels",
            description=(
                "Compare model alarms to ESA ground truth on the same skip_rows/max_rows slice. "
                "Returns ESA-ADB ESAScores plus a short true/wrong/miss note. "
                "Scores automatically when needed. Use ONLY for labels / precision / "
                "recall / known events — do not also call score_telemetry in the same turn."
            ),
            args_schema=CompareArgs,
        ),
    ]
