#!/usr/bin/env python3
"""Step 4 — Single agent with THREE tools:

  1) retrieve_knowledge       — FAISS RAG (Step 1)
  2) score_telemetry          — Telemanom model/ + scoring/ (Step 3)
  3) compare_alarms_to_labels — alarms vs is_anomaly_* ground truth

  conda activate esa-agent
  cd esa-telemanom-agent/step4_agent

  # RAG
  python agent.py "What is the window size?"

  # Score + compare to labels
  python agent.py "Score the slice and compare alarms to ground-truth labels" \\
    --csv ../../data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.test.csv \\
    --skip-rows 33064 --max-rows 3000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
DEFAULT_OUT = ROOT / "step4_agent" / "out"
MAX_RETRIEVE_K = 8


def load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv as _load
    except ImportError:
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

# MiniLM / transformers should use PyTorch only. If they also import TensorFlow,
# Streamlit (and some envs) can crash with a TF DType double-register error.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

sys.path.insert(0, str(ROOT / "step1_rag"))
sys.path.insert(0, str(ROOT))
from query import retrieve  # noqa: E402


def _scoring():
    """Import scoring only when score tools run (loads TensorFlow)."""
    from scoring import TelemanomInferencer, write_outputs

    return TelemanomInferencer, write_outputs

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-20b"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"


def system_prompt(
    csv_hint: str | None,
    default_max_rows: int = 3000,
    default_skip_rows: int = 0,
) -> str:
    base = """You are a careful assistant for the ESA Telemanom-ESA Mission1 lab project.

You have three tools:
1) retrieve_knowledge — docs/facts (channels, model card, lab metrics).
2) score_telemetry — run Telemanom on a CSV; returns a SHORT alarm summary (writes scores/alarms under step4_agent/out/).
3) compare_alarms_to_labels — compare alarms to ESA ground truth on the same CSV slice.
   Use AFTER scoring (or with re_score=true).
   Primary report: ESA-ADB ESAScores (alarming_precision, EW_precision/recall/F0.50, AFF_*),
   same definitions as the lab TimeEval Telemanom evaluation.
   Also includes a short interval true/wrong/miss note for intuition.

Rules:
- Answer ONLY from tool results. Do not invent physical root causes.
- Keep answers short (2-6 sentences). Cite tool results.
- For retrieve_knowledge, keep k between 1 and 8.
- Prefer model_card for aggregate facts (87 inputs / 58 targets).
- When the user asks precision / recall / true vs false / labels, call compare_alarms_to_labels.
- Prefer quoting ESAScores (EW_*, AFF_*, alarming_precision, F_0.50) from the tool.
  Do not invent metrics or recompute true/labels as recall.
"""
    if csv_hint:
        base += f"\nDefault telemetry CSV (has channels + is_anomaly_* labels): {csv_hint}\n"
        base += (
            f"Default scoring/compare args: csv_path={csv_hint!r}, "
            f"max_rows={default_max_rows}, skip_rows={default_skip_rows}\n"
        )
    return base


def resolve_llm_config(model_override: str | None = None) -> tuple[str, str, str]:
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
        "No LLM API key. Set GROQ_API_KEY in esa-telemanom-agent/.env\n"
        "  cp ../.env.example ../.env"
    )


def has_llm_credentials() -> bool:
    return bool(
        os.environ.get("GROQ_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_BASE_URL")
    )


def tool_retrieve_knowledge(query: str, k: int = 5) -> str:
    k = max(1, min(int(k), MAX_RETRIEVE_K))
    hits = retrieve(query, k=k)
    if not hits:
        return "No chunks found."
    blocks = []
    for i, (score, chunk) in enumerate(hits, 1):
        blocks.append(
            f"[{i}] score={score:.3f} source={chunk['source']} id={chunk['id']}\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(blocks)


def _summarize_result(result, window_size: int, top_n: int = 10) -> str:
    lines = [
        f"rows={len(result.scores)}  target_channels={result.scores.shape[1]}",
        f"window_size={window_size} (first window rows have zero scores/alarms)",
    ]
    if result.alarms is None:
        lines.append("alarms: not computed")
        return "\n".join(lines)
    counts = result.alarms.sum(axis=0).sort_values(ascending=False)
    active = counts[counts > 0]
    lines.append(f"channels_with_any_alarm={len(active)} / {len(counts)}")
    if len(active) == 0:
        lines.append("No alarms in this slice.")
    else:
        lines.append(f"Top {min(top_n, len(active))} channels by alarm timesteps:")
        for ch, cnt in active.head(top_n).items():
            lines.append(f"  {ch}: {int(cnt)}")
    return "\n".join(lines)


def tool_score_telemetry(
    csv_path: str,
    max_rows: int | None = 3000,
    skip_rows: int = 0,
    threshold: bool = True,
    output_dir: str | None = None,
) -> str:
    path = Path(csv_path)
    if not path.is_file():
        return f"Error: CSV not found: {path}"
    if not (MODEL_DIR / "model.h5").is_file():
        return f"Error: model pack missing under {MODEL_DIR}"

    out = Path(output_dir) if output_dir else DEFAULT_OUT
    try:
        TelemanomInferencer, write_outputs = _scoring()
        clf = TelemanomInferencer(artifacts_dir=MODEL_DIR)
        result = clf.predict(
            path,
            threshold=threshold,
            max_rows=max_rows,
            skip_rows=skip_rows,
        )
        write_outputs(result, out)
        summary = _summarize_result(result, window_size=int(clf.contract["window_size"]))
        return (
            f"Scored OK. Wrote {out}/scores.csv"
            + (f" and {out}/alarms.csv" if result.alarms is not None else "")
            + f"\n\n{summary}"
        )
    except Exception as e:
        return f"Error during scoring: {type(e).__name__}: {e}"


def tool_compare_alarms_to_labels(
    csv_path: str,
    max_rows: int | None = 3000,
    skip_rows: int = 0,
    re_score: bool = True,
    alarms_csv: str | None = None,
) -> str:
    """Compare alarms to is_anomaly_* labels on the same CSV slice."""
    path = Path(csv_path)
    if not path.is_file():
        return f"Error: CSV not found: {path}"

    alarms_path = Path(alarms_csv) if alarms_csv else (DEFAULT_OUT / "alarms.csv")

    try:
        if re_score or not alarms_path.is_file():
            score_msg = tool_score_telemetry(
                csv_path=str(path),
                max_rows=max_rows,
                skip_rows=skip_rows,
                threshold=True,
            )
            if score_msg.startswith("Error"):
                return score_msg
            alarms_path = DEFAULT_OUT / "alarms.csv"

        if not alarms_path.is_file():
            return f"Error: alarms file missing at {alarms_path}. Run score_telemetry first."

        from scoring import (
            compare_alarms_to_labels,
            load_labels_aligned,
            compute_esa_scores,
            format_esa_scores_report,
        )

        alarms = pd.read_csv(alarms_path, index_col=0, parse_dates=True)
        channels = list(alarms.columns)
        labels = load_labels_aligned(
            path,
            timestamps=alarms.index,
            channels=channels,
            max_rows=max_rows,
            skip_rows=skip_rows,
        )
        if labels.shape[0] != alarms.shape[0]:
            return (
                f"Error: length mismatch alarms={alarms.shape[0]} labels={labels.shape[0]}. "
                "Use the same skip_rows/max_rows as when scoring."
            )

        # Primary: ESA-ADB ESAScores (lab-comparable definitions).
        try:
            esa = compute_esa_scores(alarms)
            esa_text = format_esa_scores_report(esa)
        except Exception as e:
            esa_text = f"ESAScores unavailable ({type(e).__name__}: {e})."

        # Secondary: simple interval overlap (intuition only — not the lab metric).
        simple = compare_alarms_to_labels(alarms, labels)

        return (
            f"Compared alarms ({alarms_path}) on slice "
            f"(skip_rows={skip_rows}, max_rows={max_rows}).\n\n"
            f"{esa_text}\n\n"
            f"---\nSimple interval overlap (not ESAScores; for intuition only):\n{simple}"
        )
    except Exception as e:
        return f"Error during compare: {type(e).__name__}: {e}"


TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge",
            "description": (
                "Search lab knowledge (channels, anomaly types, model card, lab results). "
                "Keep k between 1 and 8."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_telemetry",
            "description": (
                "Run Telemanom-ESA on a telemetry CSV. Returns a short alarm summary. "
                "Writes step4_agent/out/scores.csv and alarms.csv."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string"},
                    "max_rows": {"type": "integer", "default": 3000},
                    "skip_rows": {"type": "integer", "default": 0},
                    "threshold": {"type": "boolean", "default": True},
                },
                "required": ["csv_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_alarms_to_labels",
            "description": (
                "Compare model alarms to ESA ground truth on the same skip_rows/max_rows slice. "
                "Returns ESA-ADB ESAScores: alarming_precision, EW_precision, EW_recall, EW_F_0.50, "
                "AFF_precision, AFF_recall, AFF_F_0.50 (same definitions as the lab TimeEval run), "
                "plus a short true/wrong/miss overlap note. "
                "Set re_score=true (default) to score first; set false to reuse existing out/alarms.csv."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string"},
                    "max_rows": {"type": "integer", "default": 3000},
                    "skip_rows": {"type": "integer", "default": 0},
                    "re_score": {
                        "type": "boolean",
                        "default": True,
                        "description": "If true, run score_telemetry before compare",
                    },
                },
                "required": ["csv_path"],
            },
        },
    },
]


def dispatch_tool(name: str, args: dict, default_k: int) -> str:
    if name == "retrieve_knowledge":
        return tool_retrieve_knowledge(
            args.get("query", ""),
            k=int(args.get("k", default_k)),
        )
    if name == "score_telemetry":
        return tool_score_telemetry(
            csv_path=str(args.get("csv_path", "")),
            max_rows=args.get("max_rows", 3000),
            skip_rows=int(args.get("skip_rows", 0)),
            threshold=bool(args.get("threshold", True)),
        )
    if name == "compare_alarms_to_labels":
        return tool_compare_alarms_to_labels(
            csv_path=str(args.get("csv_path", "")),
            max_rows=args.get("max_rows", 3000),
            skip_rows=int(args.get("skip_rows", 0)),
            re_score=bool(args.get("re_score", True)),
        )
    return f"Unknown tool: {name}"


def run_llm_collect(
    question: str,
    k: int,
    model: str | None,
    csv_hint: str | None,
    default_max_rows: int = 3000,
    default_skip_rows: int = 0,
) -> dict:
    """Run the ReAct tool loop and return structured results for the UI/CLI.

    Returns:
      provider, model, tool_steps (list of {tool, args, observation}), final_answer
    """
    from openai import OpenAI

    api_key, base_url, model_name = resolve_llm_config(model)
    client = OpenAI(api_key=api_key, base_url=base_url)

    if "groq.com" in base_url:
        provider = "Groq"
    elif "googleapis.com" in base_url:
        provider = "Gemini"
    else:
        provider = base_url

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

    # Max 8 rounds so a stuck loop cannot run forever.
    for _ in range(8):
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=TOOLS_SPEC,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        messages.append(msg)

        # If the model wants tools, run each one and feed results back.
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


def run_llm(
    question: str,
    k: int,
    model: str | None,
    csv_hint: str | None,
    default_max_rows: int = 3000,
    default_skip_rows: int = 0,
) -> None:
    """CLI wrapper: print what run_llm_collect returns."""
    result = run_llm_collect(
        question,
        k=k,
        model=model,
        csv_hint=csv_hint,
        default_max_rows=default_max_rows,
        default_skip_rows=default_skip_rows,
    )

    print("=== Mode: LLM agent (3 tools) ===")
    print(f"Provider: {result['provider']}")
    print(f"Model:    {result['model']}")
    if csv_hint:
        print(f"CSV hint: {csv_hint}")
    print(f"\nQuestion: {question}\n")

    for step in result["tool_steps"]:
        print(f"Thought/Action: `{step['tool']}` {step['args']}")
        obs = step["observation"]
        preview = obs if len(obs) <= 1800 else obs[:1800] + "…"
        print("Observation:\n", preview, "\n", sep="")

    print("Final answer:")
    print(result["final_answer"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: LLM agent with retrieve + score + compare-to-labels"
    )
    parser.add_argument("question", help="User question")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--model", default=None, help="Optional LLM model override")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument("--skip-rows", type=int, default=0)
    args = parser.parse_args()

    # Always call the LLM agent (needs GROQ_API_KEY or another key in .env).
    run_llm(
        args.question,
        k=args.k,
        model=args.model,
        csv_hint=args.csv,
        default_max_rows=args.max_rows,
        default_skip_rows=args.skip_rows,
    )


if __name__ == "__main__":
    main()
