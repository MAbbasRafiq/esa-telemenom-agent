"""Shared fixtures for tool regression tests.

Imports go through agent/config so paths and TRANSFORMERS_NO_TF are set
the same way as the CLI / API.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Project root = parent of tests/
ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agent"

# Same import path style as running from agent/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

# Load config first (sys.path, .env, TF flags).
import config  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to esa-telemanom-agent/."""
    return ROOT


@pytest.fixture(scope="session")
def demo_sample_csv(project_root: Path) -> Path:
    """Prepared demo window (copy of a slice; original full test file untouched)."""
    path = project_root / "samples" / "demo_known_events.csv"
    if not path.is_file():
        pytest.skip(f"Missing sample CSV: {path}")
    return path


@pytest.fixture(scope="session")
def model_pack_ok(project_root: Path) -> bool:
    """True when lab model.h5 is present."""
    return (project_root / "model" / "model.h5").is_file()


@pytest.fixture(scope="session")
def faiss_index_ok(project_root: Path) -> bool:
    """True when FAISS index was built."""
    return (project_root / "rag" / "index" / "faiss.index").is_file()
