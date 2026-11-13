"""LLM provider setup and the tool-calling agent loop.

Uses an OpenAI-compatible client (Groq / Gemini / OpenAI / Ollama).
"""
from __future__ import annotations

import json
import os

# Import config first (sets sys.path + .env).
import config
from tools import TOOLS_SPEC, dispatch_tool

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-20b"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"


def system_prompt(
    csv_hint: str | None,
    default_max_rows: int = 3000,
    default_skip_rows: int = 0,
) -> str:
    """Build the system message that tells the LLM how to use the three tools."""
    base = """You are a careful assistant for the ESA Telemanom-ESA Mission1 lab project.

You have three tools:
1) retrieve_knowledge — docs/facts (channels, model card, lab metrics).
2) score_telemetry — run Telemanom on a CSV; returns a SHORT alarm-channel summary
   (which channels alarm, top counts). No ESAScores / label metrics.
3) compare_alarms_to_labels — ESA ground-truth compare + ESAScores on the same CSV.
   Scores automatically when needed. Use ONLY when the user wants labels / precision /
   recall / true vs false / known events / ESAScores.

Tool pick (call EXACTLY ONE scoring-related tool per question — never both):
- "score / alarms / which channels" (no labels/metrics) → score_telemetry ONLY.
- "compare / precision / recall / labels / known events / ESAScores" → compare_alarms_to_labels ONLY.
- "score and compare" → compare_alarms_to_labels ONLY (it scores inside if needed).
Never call score_telemetry and compare_alarms_to_labels in the same turn.

Rules:
- Answer ONLY from tool results. Do not invent physical root causes.
- Keep answers short (2-6 sentences) or a small Markdown table. Do not invent metrics.
- For retrieve_knowledge, keep k between 1 and 8.
- Prefer model_card for aggregate facts (87 inputs / 58 targets).
- When quoting ESAScores, use only values from compare_alarms_to_labels.
"""
    if csv_hint:
        base += f"\nDefault telemetry CSV (has channels + is_anomaly_* labels): {csv_hint}\n"
        base += (
            f"Default scoring/compare args: csv_path={csv_hint!r}, "
            f"max_rows={default_max_rows}, skip_rows={default_skip_rows}\n"
        )
    return base


def resolve_llm_config(model_override: str | None = None) -> tuple[str, str, str]:
    """Pick API key, base URL, and model name from environment variables.

    Priority:
      1. GROQ_API_KEY
      2. GEMINI_API_KEY or GOOGLE_API_KEY
      3. OPENAI_API_KEY / OPENAI_BASE_URL (also works for local Ollama)
    """
    if os.environ.get("GROQ_API_KEY"):
        model_name = (
            model_override
            or os.environ.get("GROQ_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or GROQ_DEFAULT_MODEL
        )
        return os.environ["GROQ_API_KEY"], GROQ_BASE_URL, model_name

    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        model_name = (
            model_override
            or os.environ.get("GEMINI_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or GEMINI_DEFAULT_MODEL
        )
        return gemini_key, GEMINI_BASE_URL, model_name

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if api_key or base_url:
        if not api_key and base_url:
            # Local Ollama often needs any non-empty key string.
            api_key = "ollama"
        model_name = model_override or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        return api_key, base_url or "https://api.openai.com/v1", model_name

    raise SystemExit(
        "No LLM API key. Set GROQ_API_KEY in esa-telemanom-agent/.env\n"
        "  cp .env.example .env"
    )


def provider_name_from_base_url(base_url: str) -> str:
    """Return a short label for logging (Groq / Gemini / raw URL)."""
    if "groq.com" in base_url:
        return "Groq"
    if "googleapis.com" in base_url:
        return "Gemini"
    return base_url


def run_agent_legacy(
    question: str,
    k: int,
    model: str | None,
    csv_hint: str | None,
    default_max_rows: int = 3000,
    default_skip_rows: int = 0,
) -> dict:
    """Original DIY OpenAI tool-calling loop (no LangGraph / no thread memory)."""
    from openai import OpenAI

    api_key, base_url, model_name = resolve_llm_config(model)
    client = OpenAI(api_key=api_key, base_url=base_url)
    provider = provider_name_from_base_url(base_url)

    messages = [
        {
            "role": "system",
            "content": system_prompt(
                csv_hint,
                default_max_rows=default_max_rows,
                default_skip_rows=default_skip_rows,
            ),
        },
        {"role": "user", "content": question},
    ]

    tool_steps = []

    # Cap rounds so a stuck loop cannot run forever.
    max_rounds = 8
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=TOOLS_SPEC,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        messages.append(msg)

        # If the model asked for tools, run each one and append the observations.
        if msg.tool_calls:
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                result = dispatch_tool(call.function.name, args, default_k=k)
                tool_steps.append(
                    {
                        "tool": call.function.name,
                        "args": args,
                        "observation": result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )
            continue

        # No more tool calls → this message is the final natural-language answer.
        return {
            "provider": provider,
            "model": model_name,
            "tool_steps": tool_steps,
            "final_answer": (msg.content or "").strip(),
        }

    return {
        "provider": provider,
        "model": model_name,
        "tool_steps": tool_steps,
        "final_answer": "Stopped: too many tool steps.",
    }


def run_agent(
    question: str,
    k: int,
    model: str | None,
    csv_hint: str | None,
    default_max_rows: int = 3000,
    default_skip_rows: int = 0,
    thread_id: str | None = None,
) -> dict:
    """Run the agent; prefer LangGraph (session memory) when available.

    Set ESA_AGENT_LEGACY=1 to force the DIY loop (no thread history).

    Returns dict with: provider, model, tool_steps, final_answer [, thread_id]
    """
    use_legacy = os.environ.get("ESA_AGENT_LEGACY", "").strip() in ("1", "true", "yes")

    if not use_legacy:
        try:
            from graph import run_graph_agent
            from session import SESSION_STORE

            tid = thread_id or SESSION_STORE.new_thread_id()
            return run_graph_agent(
                question,
                thread_id=tid,
                k=k,
                model=model,
                csv_hint=csv_hint,
                default_max_rows=default_max_rows,
                default_skip_rows=default_skip_rows,
            )
        except ImportError:
            # LangGraph / langchain not installed — fall back.
            pass

    result = run_agent_legacy(
        question,
        k=k,
        model=model,
        csv_hint=csv_hint,
        default_max_rows=default_max_rows,
        default_skip_rows=default_skip_rows,
    )
    if thread_id:
        result["thread_id"] = thread_id
    return result
