"""Shared paths and environment setup for the agent package.

Import this module first (before rag / scoring) so:
  - the project root is on sys.path
  - transformers prefers PyTorch (avoids TensorFlow clash)
  - .env keys are loaded into os.environ
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# This folder (agent/) and the project root
AGENT_DIR = Path(__file__).resolve().parent
ROOT = AGENT_DIR.parent
MODEL_DIR = ROOT / "model"
DEFAULT_OUT = AGENT_DIR / "out"
MAX_RETRIEVE_K = 8

# Project root → "import rag", "import scoring"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Agent folder → "import config", "import tools", "import llm"
# (needed when running: cd agent && python agent.py)
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

# MiniLM / transformers should use PyTorch only (avoid TensorFlow import clash).
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")


def load_dotenv() -> None:
    """Load KEY=VALUE pairs from the project .env into os.environ (if missing)."""
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return

    try:
        from dotenv import load_dotenv as _load
    except ImportError:
        # Fallback if python-dotenv is not installed: parse the file ourselves.
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
        return

    _load(env_path, override=False)


# Load once when this module is imported.
load_dotenv()
