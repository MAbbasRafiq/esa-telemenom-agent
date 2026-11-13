"""Multi-agent LangGraph: Router → Knowledge | Score | Compare.

Each specialist gets only the tools it needs (prevents double Telemanom runs).
Shared SQLite checkpointer keeps chat memory per thread_id.
"""
from __future__ import annotations

import contextvars
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from lc_tools import build_langchain_tools
from progress import report_progress
from session import SESSION_STORE

# Set by graph.py around each invoke (CSV defaults for specialist prompts).
ACTIVE_THREAD: contextvars.ContextVar[str] = contextvars.ContextVar(
    "esa_ma_thread",
    default="",
)
ACTIVE_K: contextvars.ContextVar[int] = contextvars.ContextVar(
    "esa_ma_k",
    default=5,
)

RouteName = Literal["knowledge", "score", "compare"]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    route: str
    # Numbers in the answer that this turn's tool output does not support.
    unverified: list[str]


class RouteDecision(BaseModel):
    """Router output: which specialist should handle this user turn."""

    agent: RouteName = Field(
        description=(
            "knowledge = docs/facts only; "
            "score = Telemanom alarms on CSV (no label metrics); "
            "compare = ESAScores / labels / precision / known events"
        )
    )
    reason: str = Field(default="", description="Short why this agent")


def _session_csv_block() -> str:
    thread_id = ACTIVE_THREAD.get()
    k = ACTIVE_K.get()
    session = SESSION_STORE.get(thread_id) if thread_id else None
    csv_hint = session.csv_path if session else None
    max_rows = session.max_rows if session else 3000
    skip_rows = session.skip_rows if session else 0
    lines = [
        "Answer ONLY from tool results. Do not invent physical root causes.",
        "Prefer short answers or a small Markdown table.",
        f"Default retrieve_knowledge k={k}.",
    ]
    if csv_hint:
        lines.append(f"Default telemetry CSV: {csv_hint}")
        lines.append(
            f"Default score/compare args: csv_path={csv_hint!r}, "
            f"max_rows={max_rows}, skip_rows={skip_rows}"
        )
    else:
        lines.append("No CSV attached for this thread — do not invent a csv_path.")
    return "\n".join(lines)


def _prompt_for(role: str, role_rules: str):
    """Build a create_react_agent prompt callable for one specialist."""

    def _prompt(state: dict) -> list:
        sys = (
            f"You are the {role} specialist for ESA Telemanom-ESA Mission1.\n"
            f"{role_rules}\n\n"
            f"{_session_csv_block()}"
        )
        prior = [
            m
            for m in (state.get("messages") or [])
            if not isinstance(m, SystemMessage)
            and getattr(m, "type", None) != "system"
        ]
        return [SystemMessage(content=sys), *prior]

    return _prompt


KNOWLEDGE_RULES = """
You ONLY use retrieve_knowledge.
Answer questions about the model, window size, channels, lab docs.
Do not score telemetry or compare labels.
"""

SCORE_RULES = """
You ONLY use score_telemetry.
Use it when the user wants alarms / which channels alarm / anomaly scoring.
Do NOT call compare or invent ESAScores / precision / recall.
If no CSV path is available, say a telemetry CSV must be attached.
"""

COMPARE_RULES = """
You ONLY use compare_alarms_to_labels (it scores automatically when needed).
Use for labels, precision, recall, known events, ESAScores, true vs false alarms.
Do NOT also call score_telemetry.
If no CSV path is available, say a telemetry CSV must be attached.
"""


def heuristic_route(question: str) -> RouteName:
    """Keyword fallback if the router LLM fails or returns junk."""
    t = (question or "").lower()
    compare_keys = (
        "compare",
        "precision",
        "recall",
        "label",
        "ground truth",
        "ground-truth",
        "known event",
        "esascore",
        "esa score",
        "true vs",
        "false alarm",
        "f0.5",
        "f_0.5",
        "aff_",
        "ew_",
    )
    if any(k in t for k in compare_keys):
        return "compare"

    score_keys = (
        "score",
        "alarm",
        "anomaly",
        "anomalies",
        "telemetry",
        "detect",
        "which channel",
    )
    if any(k in t for k in score_keys):
        return "score"

    return "knowledge"


def _last_user_text(messages: Sequence[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def build_multi_agent_graph(llm: ChatOpenAI, checkpointer):
    """Compile Router → specialist graph with durable checkpoints."""
    tools = build_langchain_tools()
    by_name: dict[str, BaseTool] = {t.name: t for t in tools}

    knowledge_agent = create_react_agent(
        model=llm,
        tools=[by_name["retrieve_knowledge"]],
        prompt=_prompt_for("Knowledge", KNOWLEDGE_RULES),
        name="knowledge_agent",
    )
    score_agent = create_react_agent(
        model=llm,
        tools=[by_name["score_telemetry"]],
        prompt=_prompt_for("Score", SCORE_RULES),
        name="score_agent",
    )
    compare_agent = create_react_agent(
        model=llm,
        tools=[by_name["compare_alarms_to_labels"]],
        prompt=_prompt_for("Compare", COMPARE_RULES),
        name="compare_agent",
    )

    router_llm = llm.with_structured_output(RouteDecision)

    def router_node(state: AgentState) -> dict:
        report_progress("Router: choosing specialist…")
        question = _last_user_text(state.get("messages") or [])
        try:
            decision = router_llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "Route the user question to exactly one specialist.\n"
                            "- knowledge: model/docs/facts (window size, channels, how Telemanom works)\n"
                            "- score: run Telemanom / list alarm channels (no label metrics)\n"
                            "- compare: precision, recall, labels, known events, ESAScores\n"
                            "If the user asks to score AND compare, choose compare."
                        )
                    ),
                    HumanMessage(content=question or "Hello"),
                ]
            )
            agent: RouteName = decision.agent  # type: ignore[assignment]
            reason = (decision.reason or "").strip()
        except Exception:
            agent = heuristic_route(question)
            reason = "heuristic fallback"

        if agent not in ("knowledge", "score", "compare"):
            agent = heuristic_route(question)
            reason = "normalized route"

        label = {
            "knowledge": "Knowledge agent",
            "score": "Score agent",
            "compare": "Compare agent",
        }[agent]
        report_progress(f"Router → {label}" + (f" ({reason})" if reason else ""))
        # Clear the previous turn's flags (state is checkpointed per thread).
        return {"route": agent, "unverified": []}

    def verifier_node(state: AgentState) -> dict:
        from verify import verify_messages

        messages = list(state.get("messages") or [])
        flagged = verify_messages(messages)
        if flagged:
            report_progress(f"Check: {len(flagged)} number(s) not found in tool output")
        else:
            report_progress("Check: answer numbers match tool output")
        return {"unverified": flagged}

    def route_edge(state: AgentState) -> str:
        route = state.get("route") or "knowledge"
        if route not in ("knowledge", "score", "compare"):
            return "knowledge"
        return route

    builder = StateGraph(AgentState)
    builder.add_node("router", router_node)
    builder.add_node("knowledge", knowledge_agent)
    builder.add_node("score", score_agent)
    builder.add_node("compare", compare_agent)
    builder.add_node("verifier", verifier_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_edge,
        {
            "knowledge": "knowledge",
            "score": "score",
            "compare": "compare",
        },
    )
    builder.add_edge("knowledge", "verifier")
    builder.add_edge("score", "verifier")
    builder.add_edge("compare", "verifier")
    builder.add_edge("verifier", END)

    return builder.compile(checkpointer=checkpointer)


def describe_route(route: str) -> str:
    """Human label for UI / progress."""
    return {
        "knowledge": "Knowledge agent",
        "score": "Score agent",
        "compare": "Compare agent",
    }.get(route, route or "agent")
