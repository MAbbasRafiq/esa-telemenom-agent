"""Agent tools: retrieve knowledge, score telemetry, compare alarms to labels.

Each tool returns a plain string (for the LLM observation).
TOOLS_SPEC describes the same tools in OpenAI function-calling format.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Import config first (sets sys.path, .env, TRANSFORMERS_NO_TF).
import config
from config import DEFAULT_OUT, MAX_RETRIEVE_K, MODEL_DIR

from rag.query import retrieve


def _load_scoring():
    """Import scoring helpers only when score tools run (loads TensorFlow)."""
    from scoring import TelemanomInferencer, write_outputs

    return TelemanomInferencer, write_outputs


def tool_retrieve_knowledge(query: str, k: int = 5) -> str:
    """Search the FAISS knowledge index and return a readable text block."""
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
    """Build a short human-readable summary of a Telemanom scoring result."""
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
    """Run Telemanom on a CSV slice; write scores/alarms; return a short summary."""
    path = Path(csv_path)
    if not path.is_file():
        return f"Error: CSV not found: {path}"
    if not (MODEL_DIR / "model.h5").is_file():
        return f"Error: model pack missing under {MODEL_DIR}"

    out = Path(output_dir) if output_dir else DEFAULT_OUT

    try:
        TelemanomInferencer, write_outputs = _load_scoring()
        clf = TelemanomInferencer(artifacts_dir=MODEL_DIR)
        result = clf.predict(
            path,
            threshold=threshold,
            max_rows=max_rows,
            skip_rows=skip_rows,
        )
        write_outputs(result, out)
        summary = _summarize_result(result, window_size=int(clf.contract["window_size"]))
        alarm_note = f" and {out}/alarms.csv" if result.alarms is not None else ""
        return f"Scored OK. Wrote {out}/scores.csv{alarm_note}\n\n{summary}"
    except Exception as e:
        return f"Error during scoring: {type(e).__name__}: {e}"


def tool_compare_alarms_to_labels(
    csv_path: str,
    max_rows: int | None = 3000,
    skip_rows: int = 0,
    re_score: bool = True,
    alarms_csv: str | None = None,
) -> str:
    """Compare model alarms to is_anomaly_* labels on the same CSV slice.

    Primary metrics come from ESA-ADB ESAScores (same as lab TimeEval).
    Secondary note is simple interval overlap (true / wrong / miss).
    """
    path = Path(csv_path)
    if not path.is_file():
        return f"Error: CSV not found: {path}"

    alarms_path = Path(alarms_csv) if alarms_csv else (DEFAULT_OUT / "alarms.csv")

    try:
        # Score first unless the caller asks to reuse an existing alarms file.
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
            return (
                f"Error: alarms file missing at {alarms_path}. "
                "Run score_telemetry first."
            )

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
                f"Error: length mismatch alarms={alarms.shape[0]} "
                f"labels={labels.shape[0]}. "
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


# Tool schemas sent to the LLM (OpenAI-compatible function-calling format).
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
                "Writes agent/out/scores.csv and alarms.csv."
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
    """Call the Python function that matches a tool name from the LLM."""
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
