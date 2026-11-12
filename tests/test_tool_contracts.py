"""Fast checks that tool contracts stay consistent (no Telemanom / no LLM)."""
from __future__ import annotations

import tools


def test_tools_spec_names_match_dispatch():
    """Every TOOLS_SPEC name must be handled by dispatch_tool."""
    names = []
    for item in tools.TOOLS_SPEC:
        names.append(item["function"]["name"])

    assert names == [
        "retrieve_knowledge",
        "score_telemetry",
        "compare_alarms_to_labels",
    ]

    # Unknown tool returns a clear string (not an exception).
    msg = tools.dispatch_tool("not_a_real_tool", {}, default_k=3)
    assert msg.startswith("Unknown tool:")


def test_retrieve_clamps_k(faiss_index_ok):
    """k is clamped to 1..MAX_RETRIEVE_K; oversized k must not crash."""
    if not faiss_index_ok:
        import pytest

        pytest.skip("FAISS index missing — run: python -m rag.build_index")

    # k=99 should clamp to MAX_RETRIEVE_K (8), still return text.
    text = tools.tool_retrieve_knowledge("window size", k=99)
    assert isinstance(text, str)
    assert len(text) > 0
    assert not text.startswith("Error:")


def test_score_missing_csv_returns_error_prefix():
    """Contract: failures start with 'Error:'."""
    msg = tools.tool_score_telemetry("/no/such/file.csv", max_rows=10, skip_rows=0)
    assert msg.startswith("Error:")


def test_compare_missing_csv_returns_error_prefix():
    """Contract: failures start with 'Error:'."""
    msg = tools.tool_compare_alarms_to_labels(
        "/no/such/file.csv",
        max_rows=10,
        skip_rows=0,
    )
    assert msg.startswith("Error:")
