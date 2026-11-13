"""Multi-agent routing regression (no LLM, no Telemanom).

Guards the Router contract: one specialist per turn, and each specialist
only owns its tool.
"""
from __future__ import annotations

import pytest

import multi_agent


# (question, expected agent)
ROUTING_CASES = [
    # Knowledge: docs / model facts
    ("What is the look-back window size?", "knowledge"),
    ("How many input and target channels does the model use?", "knowledge"),
    ("How does Telemanom smooth errors?", "knowledge"),
    # Score: run detection, list alarm channels
    ("Score the attached telemetry and summarize which channels show alarms.", "score"),
    ("Find anomalies in this file", "score"),
    ("Which channels alarm in this window?", "score"),
    # Compare: labels / metrics
    ("Compare alarms to ground-truth labels", "compare"),
    ("What is EW precision and recall for this slice?", "compare"),
    ("How well do detections match known events?", "compare"),
    ("Show ESAScores for the attached CSV", "compare"),
]


@pytest.mark.parametrize("question,expected", ROUTING_CASES)
def test_heuristic_route(question: str, expected: str):
    """Keyword fallback must agree with the intended specialist."""
    assert multi_agent.heuristic_route(question) == expected


def test_heuristic_route_defaults_to_knowledge():
    """Vague questions go to the docs specialist, not scoring."""
    assert multi_agent.heuristic_route("") == "knowledge"
    assert multi_agent.heuristic_route("hello") == "knowledge"
    assert multi_agent.heuristic_route("tell me about this project") == "knowledge"


def test_score_and_compare_prefers_compare():
    """'score and compare' must not trigger two Telemanom passes."""
    assert (
        multi_agent.heuristic_route(
            "Score the telemetry and compare alarms to labels"
        )
        == "compare"
    )


def test_specialist_rules_are_single_tool():
    """Each specialist prompt forbids the other scoring tool."""
    knowledge_rules = multi_agent.KNOWLEDGE_RULES.lower()
    assert "retrieve_knowledge" in knowledge_rules
    assert "do not score telemetry" in knowledge_rules

    score_rules = multi_agent.SCORE_RULES.lower()
    assert "score_telemetry" in score_rules
    assert "do not call compare" in score_rules

    compare_rules = multi_agent.COMPARE_RULES.lower()
    assert "compare_alarms_to_labels" in compare_rules
    assert "do not also call score_telemetry" in compare_rules


def test_route_labels_are_human_readable():
    assert multi_agent.describe_route("knowledge") == "Knowledge agent"
    assert multi_agent.describe_route("score") == "Score agent"
    assert multi_agent.describe_route("compare") == "Compare agent"


def test_specialists_own_exactly_one_tool():
    """Tool wiring check: build the tool set the graph hands to each agent."""
    from lc_tools import build_langchain_tools

    names = sorted(t.name for t in build_langchain_tools())
    assert names == [
        "compare_alarms_to_labels",
        "retrieve_knowledge",
        "score_telemetry",
    ]


def test_resolve_slice_pins_session_values():
    """Score and compare must agree on the slice so alarms are reused."""
    from lc_tools import resolve_slice
    from session import SESSION_STORE

    thread_id = "routing_slice_test"
    SESSION_STORE.upsert(
        thread_id,
        csv_path="/tmp/demo_slice.csv",
        max_rows=20000,
        skip_rows=0,
    )
    token = multi_agent.ACTIVE_THREAD.set(thread_id)
    try:
        # Score agent asks with its schema default…
        score_slice = resolve_slice("/tmp/demo_slice.csv", 3000, 0)
        # …compare agent asks with a different one.
        compare_slice = resolve_slice("/tmp/demo_slice.csv", 5000, 10)
    finally:
        multi_agent.ACTIVE_THREAD.reset(token)
        SESSION_STORE.reset(thread_id)

    assert score_slice == compare_slice
    assert score_slice == ("/tmp/demo_slice.csv", 20000, 0)
