#!/usr/bin/env python3
"""
Streamlit UI for the ESA Telemanom single agent (Step 4).

What this page does:
  - Lets you type a question
  - Optionally set a telemetry CSV slice (for score / label compare)
  - Runs the same LLM agent as agent.py (Groq tool-calling, 3 tools)
  - Shows the final answer + each tool call in plain language

Run (from this folder, with conda env esa-agent):
  streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# BEFORE Streamlit / TensorFlow / transformers imports:
# - Keep sentence-transformers on PyTorch only (do not pull TF for MiniLM).
# - Avoid Streamlit file-watcher reloading TF twice (DType register crash).
# ---------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import streamlit as st

# ---------------------------------------------------------------------------
# Paths: make sure we can import agent.py and project packages
# ---------------------------------------------------------------------------
STEP4_DIR = Path(__file__).resolve().parent
ROOT = STEP4_DIR.parent

# Put step4 + project root on sys.path (simple, explicit — no fancy tricks)
if str(STEP4_DIR) not in sys.path:
    sys.path.insert(0, str(STEP4_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def get_agent_module():
    """Import agent only when needed (not at page load).

    Why: importing agent pulls FAISS + (later) TensorFlow. Doing that at the
    top of a Streamlit script often crashes on reload with a TF DType error.
    """
    import agent as agent_mod

    return agent_mod


# Default CSV used in demos (relative to step4_agent/). Exists on this lab PC.
DEFAULT_CSV = (
    ROOT.parent
    / "data"
    / "preprocessed"
    / "multivariate"
    / "ESA-Mission1-semi-supervised"
    / "84_months.test.csv"
)

# Ready-made questions so you do not have to invent demos every time.
EXAMPLE_QUESTIONS = {
    "RAG — window size": "What is the look-back window size for Telemanom-ESA?",
    "RAG — inputs/targets": "How many input channels and target channels does the model use?",
    "Score + compare (labelled slice)": (
        "Score this slice and compare alarms to ground-truth labels. "
        "Summarize true vs wrong vs miss and precision/recall. "
        "Prefer re_score=false on compare if you already scored."
    ),
    "Score only": "Score this telemetry slice. How many channels alarmed and what are the top 5?",
}


# ---------------------------------------------------------------------------
# Small helpers (one job each)
# ---------------------------------------------------------------------------

def page_setup() -> None:
    """Configure Streamlit page title and wide layout."""
    st.set_page_config(
        page_title="ESA Telemanom Agent",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_styles() -> None:
    """High-contrast light UI (teal accent). Forces readable Streamlit labels/buttons."""
    st.markdown(
        """
        <style>
        /* App background — soft teal wash, not flat white */
        .stApp {
            background: linear-gradient(165deg, #eef4f5 0%, #f7fafb 50%, #e8f0f1 100%);
            color: #14232b;
        }

        /* Sidebar: light panel + dark text (fixes invisible labels) */
        [data-testid="stSidebar"] {
            background: #dfe9eb !important;
            border-right: 1px solid #b7c8cc;
        }
        [data-testid="stSidebar"] * {
            color: #14232b !important;
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: #14232b !important;
        }

        /* Main area headings + labels */
        h1, h2, h3, .stMarkdown, label, p {
            color: #14232b !important;
        }

        /* Example question buttons: dark text on white */
        .stButton > button {
            background: #ffffff !important;
            color: #14232b !important;
            border: 1px solid #9bb0b8 !important;
            font-weight: 600 !important;
        }
        .stButton > button:hover {
            background: #e6f3f2 !important;
            border-color: #0d7377 !important;
            color: #0a5558 !important;
        }
        /* Primary Run button */
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"] {
            background: #0d7377 !important;
            color: #ffffff !important;
            border: 1px solid #0a5558 !important;
        }

        /* Text inputs / text areas: clear black on white */
        .stTextInput input, .stTextArea textarea, .stNumberInput input {
            background: #ffffff !important;
            color: #14232b !important;
            border: 1px solid #9bb0b8 !important;
        }

        .esa-hero {
            padding: 0.4rem 0 1.1rem 0;
            border-bottom: 2px solid #0d7377;
            margin-bottom: 1.1rem;
        }
        .esa-brand {
            font-size: 1.85rem;
            font-weight: 700;
            color: #0a3d40 !important;
            margin: 0;
        }
        .esa-sub {
            color: #3d5560 !important;
            margin: 0.35rem 0 0 0;
            font-size: 0.95rem;
        }
        .esa-answer {
            background: #ffffff;
            border-left: 5px solid #0d7377;
            padding: 1rem 1.15rem;
            border-radius: 0 8px 8px 0;
            color: #14232b;
            white-space: pre-wrap;
            line-height: 1.55;
            border: 1px solid #c5d4d8;
            border-left-width: 5px;
        }
        .esa-tool-name {
            color: #0d7377;
            font-weight: 700;
            font-family: ui-monospace, monospace;
        }
        .esa-pill {
            display: inline-block;
            background: #d4efec;
            color: #0a5558;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.4rem;
            border: 1px solid #8fc9c3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_hero() -> None:
    """Top banner: project name + one short line."""
    st.markdown(
        """
        <div class="esa-hero">
          <p class="esa-brand">ESA Telemanom Agent</p>
          <p class="esa-sub">
            Single agent · 3 tools (retrieve · score · compare) · no LangChain
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _status_row(ok: bool, label: str) -> None:
    """Sidebar status line with high-contrast OK / missing text."""
    if ok:
        st.sidebar.markdown(f"**OK** — {label}")
    else:
        st.sidebar.markdown(f"**Missing** — {label}")


def check_project_ready() -> dict:
    """Check files the UI needs. Returns a dict of bool flags for the sidebar."""
    index_dir = ROOT / "step1_rag" / "index"
    # Light check for API key without importing the full agent (avoids TF at startup).
    has_key = bool(
        os.environ.get("GROQ_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_BASE_URL")
    )
    # Also read .env lightly if key not already in the process env.
    if not has_key:
        env_path = ROOT / ".env"
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key in (
                    "GROQ_API_KEY",
                    "GEMINI_API_KEY",
                    "GOOGLE_API_KEY",
                    "OPENAI_API_KEY",
                    "OPENAI_BASE_URL",
                ) and val:
                    has_key = True
                    break

    return {
        "env_file": (ROOT / ".env").is_file(),
        "has_llm_key": has_key,
        "model_h5": (ROOT / "model" / "model.h5").is_file(),
        "faiss_index": (index_dir / "faiss.index").is_file(),
        "default_csv": DEFAULT_CSV.is_file(),
    }


def show_sidebar_status(ready: dict) -> None:
    """Sidebar checklist so you see what is missing before you click Run."""
    st.sidebar.markdown("### Project status")
    _status_row(ready["faiss_index"], "FAISS index (Step 1)")
    _status_row(ready["model_h5"], "Telemanom model.h5")
    _status_row(ready["has_llm_key"], "LLM API key (.env)")
    _status_row(ready["default_csv"], "Demo CSV on disk")


def build_sidebar_controls(ready: dict) -> dict:
    """Sidebar form: CSV slice + RAG k. Always uses the LLM agent."""
    st.sidebar.markdown("### Settings")
    st.sidebar.caption("Always uses the LLM agent (needs GROQ_API_KEY in .env).")

    st.sidebar.markdown("#### Telemetry slice (for score / compare)")
    use_csv = st.sidebar.checkbox(
        "Attach CSV path",
        value=ready["default_csv"],
        help="Needed when you ask to score or compare to labels.",
    )

    if use_csv:
        default_path = str(DEFAULT_CSV) if ready["default_csv"] else ""
        csv_path = st.sidebar.text_input("CSV path", value=default_path)
        # Labelled demo block starts near row 33564 — defaults cover that.
        # Demo slice where Telemanom often overlaps GT labels (true>0).
        skip_rows = st.sidebar.number_input(
            "skip_rows",
            min_value=0,
            value=295000,
            step=100,
            help="How many CSV rows to skip from the start.",
        )
        max_rows = st.sidebar.number_input(
            "max_rows",
            min_value=500,
            value=3500,
            step=100,
            help="Prefer ~2350+ for stable Telemanom thresholds.",
        )
    else:
        csv_path = None
        skip_rows = 0
        max_rows = 3000

    k = st.sidebar.slider(
        "RAG top-k",
        min_value=1,
        max_value=8,
        value=5,
        help="How many knowledge chunks to retrieve (capped at 8).",
    )

    return {
        "csv_path": csv_path.strip() if csv_path else None,
        "skip_rows": int(skip_rows),
        "max_rows": int(max_rows),
        "k": int(k),
    }


def show_example_buttons() -> None:
    """Clickable example questions that fill the text box via session_state."""
    st.markdown("**Example questions**")
    cols = st.columns(2)
    keys = list(EXAMPLE_QUESTIONS.keys())
    for i, key in enumerate(keys):
        col = cols[i % 2]
        if col.button(key, use_container_width=True):
            st.session_state["question"] = EXAMPLE_QUESTIONS[key]


def run_agent(question: str, settings: dict) -> dict:
    """Call the LLM agent (same as CLI). Returns the result dict."""
    agent_mod = get_agent_module()
    # LLM picks which of the 3 tools to call.
    return agent_mod.run_llm_collect(
        question=question,
        k=settings["k"],
        model=None,
        csv_hint=settings["csv_path"],
        default_max_rows=settings["max_rows"],
        default_skip_rows=settings["skip_rows"],
    )


def show_final_answer(text: str) -> None:
    """Big readable box for the agent's final answer."""
    st.markdown("### Answer")
    # Escape HTML so tool text cannot break the page; keep newlines visible.
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    st.markdown(f'<div class="esa-answer">{safe}</div>', unsafe_allow_html=True)


def show_tool_steps(steps: list) -> None:
    """List each tool call: name, args, and observation text."""
    st.markdown("### Tool steps")
    if not steps:
        st.info("No tools were called.")
        return

    for i, step in enumerate(steps, start=1):
        tool_name = step.get("tool", "?")
        args = step.get("args", {})
        obs = step.get("observation", "")

        with st.expander(f"Step {i}: {tool_name}", expanded=(i == 1)):
            st.markdown(
                f'<span class="esa-pill">tool</span> '
                f'<span class="esa-tool-name">{tool_name}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("**Arguments**")
            st.code(json.dumps(args, indent=2), language="json")
            st.markdown("**Observation**")
            st.text(obs)


def show_meta_pills(result: dict) -> None:
    """Small line: which provider/model answered."""
    provider = result.get("provider", "?")
    model = result.get("model", "?")
    n_tools = len(result.get("tool_steps") or [])
    st.markdown(
        f'<span class="esa-pill">{provider}</span>'
        f'<span class="esa-pill">{model}</span>'
        f'<span class="esa-pill">{n_tools} tool call(s)</span>',
        unsafe_allow_html=True,
    )


def main() -> None:
    """Build the whole page: sidebar → question → run → show results."""
    page_setup()
    inject_styles()
    show_hero()

    ready = check_project_ready()
    show_sidebar_status(ready)
    settings = build_sidebar_controls(ready)

    # Warn early if API key is missing (agent cannot run without it).
    if not ready["has_llm_key"]:
        st.warning(
            "No LLM API key found. Add GROQ_API_KEY to esa-telemanom-agent/.env "
            "before running the agent."
        )

    show_example_buttons()

    # Keep the question in session_state so example buttons can fill it.
    if "question" not in st.session_state:
        st.session_state["question"] = EXAMPLE_QUESTIONS["RAG — window size"]

    question = st.text_area(
        "Your question",
        height=110,
        key="question",
        help="Ask about the model (RAG) or ask to score/compare a CSV slice.",
    )

    run_clicked = st.button("Run agent", type="primary", use_container_width=False)

    if run_clicked:
        if not question.strip():
            st.error("Please enter a question.")
            return

        # Scoring needs model + (usually) CSV; we still let the agent try.
        with st.spinner("Agent is working… (scoring can take a few minutes on CPU)"):
            try:
                result = run_agent(question.strip(), settings)
            except SystemExit as e:
                # resolve_llm_config raises SystemExit if no key
                st.error(str(e))
                return
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
                return

        show_meta_pills(result)
        st.write("")
        show_final_answer(result.get("final_answer") or "(empty answer)")
        st.write("")
        show_tool_steps(result.get("tool_steps") or [])

        # Optional: remind where CSV outputs landed after a score.
        out_alarms = STEP4_DIR / "out" / "alarms.csv"
        if out_alarms.is_file():
            st.caption(f"Latest alarms file: `{out_alarms}`")


if __name__ == "__main__":
    main()
