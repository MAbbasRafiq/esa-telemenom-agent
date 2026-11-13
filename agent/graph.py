"""LangGraph multi-agent entry: Router → Knowledge | Score | Compare.

SQLite checkpointing keeps chat memory per thread_id across restarts.
Public entry: run_graph_agent(...) → same dict shape as llm.run_agent().
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from config import CHECKPOINT_DB, DATA_DIR
from llm import provider_name_from_base_url, resolve_llm_config
from multi_agent import (
    ACTIVE_K,
    ACTIVE_THREAD,
    build_multi_agent_graph,
    describe_route,
)
from session import SESSION_STORE

# Process-wide SQLite checkpointer (kept open for the API process lifetime).
_CHECKPOINTER_CM = None
_CHECKPOINTER = None
_GRAPH = None
_GRAPH_MODEL_KEY: Optional[tuple[str, str, str]] = None


def get_checkpointer():
    """Lazy-open durable LangGraph checkpointer under data/."""
    global _CHECKPOINTER_CM, _CHECKPOINTER
    if _CHECKPOINTER is not None:
        return _CHECKPOINTER

    from langgraph.checkpoint.sqlite import SqliteSaver

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _CHECKPOINTER_CM = SqliteSaver.from_conn_string(str(CHECKPOINT_DB))
    _CHECKPOINTER = _CHECKPOINTER_CM.__enter__()
    _CHECKPOINTER.setup()
    return _CHECKPOINTER


def _get_graph(model_override: Optional[str] = None):
    """Build (or reuse) the multi-agent graph for the current LLM config."""
    global _GRAPH, _GRAPH_MODEL_KEY

    api_key, base_url, model_name = resolve_llm_config(model_override)
    key = (api_key, base_url or "", model_name)
    if _GRAPH is not None and _GRAPH_MODEL_KEY == key:
        return _GRAPH, provider_name_from_base_url(base_url or ""), model_name

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0,
    )
    _GRAPH = build_multi_agent_graph(llm, checkpointer=get_checkpointer())
    _GRAPH_MODEL_KEY = key
    return _GRAPH, provider_name_from_base_url(base_url or ""), model_name


def _extract_tool_steps(messages: list[Any]) -> list[dict]:
    """Pull tool_steps from messages after the last HumanMessage (this ask only)."""
    last_human = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            last_human = i

    relevant = messages[last_human + 1 :] if last_human >= 0 else messages

    pending: dict[str, dict] = {}
    steps: list[dict] = []

    for msg in relevant:
        if isinstance(msg, AIMessage) or getattr(msg, "type", None) == "ai":
            for call in getattr(msg, "tool_calls", None) or []:
                if isinstance(call, dict):
                    call_id = call.get("id") or ""
                    name = call.get("name") or ""
                    args = call.get("args") or {}
                else:
                    call_id = getattr(call, "id", "") or ""
                    name = getattr(call, "name", "") or ""
                    args = getattr(call, "args", {}) or {}
                pending[call_id] = {"tool": name, "args": args, "observation": ""}

        if isinstance(msg, ToolMessage) or getattr(msg, "type", None) == "tool":
            call_id = getattr(msg, "tool_call_id", "") or ""
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if call_id in pending:
                entry = pending.pop(call_id)
                entry["observation"] = content
                steps.append(entry)
            else:
                steps.append(
                    {
                        "tool": getattr(msg, "name", "") or "tool",
                        "args": {},
                        "observation": content,
                    }
                )

    steps.extend(pending.values())
    return steps


def _messages_to_turns(messages: list[Any]) -> list[dict]:
    """Group a thread's message history into (question, answer) turns."""
    turns: list[dict] = []
    current: Optional[dict] = None

    for msg in messages:
        kind = getattr(msg, "type", None)
        content = msg.content if isinstance(getattr(msg, "content", ""), str) else ""

        if kind == "human":
            current = {"question": content.strip(), "answer": "", "tools": []}
            turns.append(current)
        elif current is None:
            continue
        elif kind == "tool":
            name = getattr(msg, "name", "") or "tool"
            if name not in current["tools"]:
                current["tools"].append(name)
        elif kind == "ai" and not getattr(msg, "tool_calls", None) and content.strip():
            current["answer"] = content.strip()

    return turns


def load_thread_turns(thread_id: str) -> list[dict]:
    """Replay a saved thread for the UI, without building the LLM graph.

    Reads the checkpointer directly so listing past chats works even when no
    API key is configured.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = get_checkpointer().get_tuple(config)
    except Exception:
        return []
    if snapshot is None:
        return []

    values = snapshot.checkpoint.get("channel_values") or {}
    messages = list(values.get("messages") or [])
    return _messages_to_turns(messages)


def delete_thread(thread_id: str) -> None:
    """Forget a chat: session extras plus checkpointed history."""
    SESSION_STORE.reset(thread_id)
    try:
        checkpointer = get_checkpointer()
        deleter = getattr(checkpointer, "delete_thread", None)
        if callable(deleter):
            deleter(thread_id)
    except Exception:
        # History cleanup is best-effort; the session is already gone.
        pass


def _reused_alarms(tool_steps: list[dict]) -> Optional[bool]:
    """Did a scoring tool reuse alarms.csv instead of running Telemanom?"""
    scoring = [
        step
        for step in tool_steps
        if step.get("tool") in ("score_telemetry", "compare_alarms_to_labels")
    ]
    if not scoring:
        return None
    return any("reus" in (step.get("observation") or "").lower() for step in scoring)


def _final_answer_from_messages(messages: list[Any]) -> str:
    """Last AI text content with no pending tool calls."""
    for msg in reversed(messages):
        is_ai = isinstance(msg, AIMessage) or getattr(msg, "type", None) == "ai"
        if not is_ai:
            continue
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            continue
        content = msg.content
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()
    return ""


def run_graph_agent(
    question: str,
    *,
    thread_id: str,
    k: int = 5,
    model: Optional[str] = None,
    csv_hint: Optional[str] = None,
    default_max_rows: int = 3000,
    default_skip_rows: int = 0,
) -> dict:
    """Run one user turn on the multi-agent graph with checkpointed history.

    Returns dict: provider, model, tool_steps, final_answer, thread_id, routed_to
    """
    SESSION_STORE.upsert(
        thread_id,
        csv_path=csv_hint,
        max_rows=default_max_rows,
        skip_rows=default_skip_rows,
        title=question,
    )

    graph, provider, model_name = _get_graph(model)

    from progress import report_progress
    from runlog import log_turn

    started = time.monotonic()
    token_t = ACTIVE_THREAD.set(thread_id)
    token_k = ACTIVE_K.set(k)
    messages: list[Any] = []
    routed_to = ""
    unverified: list[str] = []
    try:
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 24,
        }
        report_progress("Starting multi-agent run…")
        for state in graph.stream(
            {"messages": [HumanMessage(content=question)]},
            config=config,
            stream_mode="values",
        ):
            if not isinstance(state, dict):
                continue
            if state.get("route"):
                routed_to = str(state["route"])
            if "unverified" in state:
                unverified = [str(x) for x in (state.get("unverified") or [])]
            messages = list(state.get("messages") or [])
            if not messages:
                continue
            last = messages[-1]
            if isinstance(last, AIMessage) or getattr(last, "type", None) == "ai":
                if getattr(last, "tool_calls", None):
                    agent_label = describe_route(routed_to) if routed_to else "Specialist"
                    report_progress(f"{agent_label}: planning tools…")
                else:
                    report_progress("Writing answer…")
            elif isinstance(last, ToolMessage) or getattr(last, "type", None) == "tool":
                report_progress("Got tool results — reasoning…")
    except Exception as exc:
        log_turn(
            thread_id=thread_id,
            question=question,
            route=routed_to or None,
            duration_ms=int((time.monotonic() - started) * 1000),
            provider=provider,
            model=model_name,
            max_rows=default_max_rows,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        ACTIVE_THREAD.reset(token_t)
        ACTIVE_K.reset(token_k)

    report_progress("Finishing answer…")

    try:
        from tools import _scoring_meta_path

        meta_path = _scoring_meta_path()
        if meta_path.is_file():
            SESSION_STORE.upsert(
                thread_id,
                scoring_meta=json.loads(meta_path.read_text(encoding="utf-8")),
            )
    except Exception:
        pass

    tool_steps = _extract_tool_steps(messages)
    final_answer = _final_answer_from_messages(messages)

    log_turn(
        thread_id=thread_id,
        question=question,
        route=routed_to or None,
        tools=[step.get("tool", "") for step in tool_steps],
        duration_ms=int((time.monotonic() - started) * 1000),
        provider=provider,
        model=model_name,
        max_rows=default_max_rows,
        reused_alarms=_reused_alarms(tool_steps),
        unverified=unverified,
        answer_chars=len(final_answer),
    )

    return {
        "provider": provider,
        "model": model_name,
        "tool_steps": tool_steps,
        "final_answer": final_answer,
        "thread_id": thread_id,
        "routed_to": routed_to or None,
        "unverified_claims": unverified,
    }
