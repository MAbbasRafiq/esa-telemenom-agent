"""RAG regression: look-back window fact is retrievable."""
from __future__ import annotations

import tools


def test_retrieve_lookback_window(faiss_index_ok):
    """Top chunks for 'look-back window' should mention 250 timesteps."""
    if not faiss_index_ok:
        import pytest

        pytest.skip("FAISS index missing — run: python -m rag.build_index")

    text = tools.tool_retrieve_knowledge("look-back window size", k=3)
    assert not text.startswith("Error:")
    assert "250" in text
    # Model card wording (tolerant of small phrasing changes).
    lower = text.lower()
    assert "window" in lower
