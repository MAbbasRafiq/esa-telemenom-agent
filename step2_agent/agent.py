#!/usr/bin/env python3
"""Step 2 — Single reasoning agent + one tool: retrieve_knowledge.

Architecture (ReAct-style):
  Question
    → Reason: do I need knowledge?
    → Act: call retrieve_knowledge (Step 1 FAISS)
    → Observe: chunks
    → Final short answer

Always uses an LLM (OpenAI-compatible clients):
  Prefer Groq (GROQ_API_KEY) — free, fast, good tool-calling.
  Else Gemini (GEMINI_API_KEY / GOOGLE_API_KEY).
  Else OPENAI_API_KEY / OPENAI_BASE_URL (paid OpenAI or local Ollama).

No Telemanom scoring here. That is Step 3 / Step 4.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv() -> None:
    """Load esa-telemanom-agent/.env into os.environ (does not override existing vars)."""
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv as _load
    except ImportError:
        # Minimal fallback if python-dotenv not installed
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
        return
    _load(env_path, override=False)


load_dotenv()

# Reuse Step 1 retrieve (same FAISS index)
STEP1 = ROOT / "step1_rag"
sys.path.insert(0, str(STEP1))
from query import retrieve  # noqa: E402


SYSTEM = """You are a careful assistant for the ESA Telemanom-ESA Mission1 lab project.
You answer ONLY from tool results (retrieved knowledge chunks).
If the chunks are not enough, say what is missing. Keep answers short (2-5 sentences).
Always mention the source file names you used (e.g. model_card.txt, channels.csv).
"""

# Default free providers (OpenAI-compatible HTTP APIs)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-20b"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"


def resolve_llm_config(model_override: str | None = None) -> tuple[str, str, str]:
    """Return (api_key, base_url, model) for a free / compatible provider.

    Priority: Groq → Gemini → OpenAI / custom OPENAI_BASE_URL.
    """
    if os.environ.get("GROQ_API_KEY"):
        return (
            os.environ["GROQ_API_KEY"],
            GROQ_BASE_URL,
            model_override
            or os.environ.get("GROQ_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or GROQ_DEFAULT_MODEL,
        )

    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        return (
            gemini_key,
            GEMINI_BASE_URL,
            model_override
            or os.environ.get("GEMINI_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or GEMINI_DEFAULT_MODEL,
        )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if api_key or base_url:
        if not api_key and base_url:
            api_key = "ollama"
        return (
            api_key,
            base_url or "https://api.openai.com/v1",
            model_override or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
        )

    raise SystemExit(
        "No LLM API key found.\n"
        "  Free (recommended): set GROQ_API_KEY in ../.env  # https://console.groq.com\n"
        "  Free alternative:   set GEMINI_API_KEY in ../.env  # https://aistudio.google.com/apikey"
    )


def tool_retrieve_knowledge(query: str, k: int = 5) -> str:
    """Tool the agent can call. Returns readable chunk text for the LLM."""
    hits = retrieve(query, k=k)
    blocks = []
    for i, (score, chunk) in enumerate(hits, 1):
        blocks.append(
            f"[{i}] score={score:.3f} source={chunk['source']} id={chunk['id']}\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(blocks) if blocks else "No chunks found."


def run_llm(question: str, k: int, model: str | None) -> None:
    """OpenAI-compatible tool-calling agent (Groq / Gemini / OpenAI / Ollama)."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit(
            "Install openai in the Step 1 venv: pip install openai\n" + str(e)
        )

    api_key, base_url, model_name = resolve_llm_config(model)
    client = OpenAI(api_key=api_key, base_url=base_url)
    provider = (
        "Groq"
        if "groq.com" in base_url
        else "Gemini"
        if "googleapis.com" in base_url
        else base_url
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "retrieve_knowledge",
                "description": (
                    "Search the ESA Telemanom lab knowledge base (channels, "
                    "anomaly types, lab run results, model card). Use before answering."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for the knowledge base",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of chunks to return",
                            "default": k,
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    print("=== Mode: LLM tool-calling agent ===")
    print(f"Provider: {provider}")
    print(f"Model:    {model_name}\n")
    print(f"Question: {question}\n")

    for step in range(6):
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                print(f"Thought/Action: call tool `{name}` with {args}")
                if name != "retrieve_knowledge":
                    result = f"Unknown tool: {name}"
                else:
                    q = args.get("query", question)
                    kk = int(args.get("k", k))
                    result = tool_retrieve_knowledge(q, k=kk)
                print("Observation (truncated):")
                print(result[:1200] + ("…" if len(result) > 1200 else ""))
                print()
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )
            continue

        answer = (msg.content or "").strip()
        print("Final answer:")
        print(answer)
        return

    print("Stopped: too many tool steps without a final answer.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: LLM agent + retrieve_knowledge tool"
    )
    parser.add_argument("question", help="User question")
    parser.add_argument("-k", type=int, default=5, help="Chunks to retrieve")
    parser.add_argument(
        "--model",
        default=None,
        help="Override chat model (else provider default / env)",
    )
    args = parser.parse_args()

    # Always call the LLM agent (needs GROQ_API_KEY or another key in .env).
    run_llm(args.question, k=args.k, model=args.model)


if __name__ == "__main__":
    main()
