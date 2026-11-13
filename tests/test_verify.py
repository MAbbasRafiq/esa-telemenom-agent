"""Numeric grounding checks (pure Python, no LLM / no Telemanom)."""
from __future__ import annotations

from verify import (
    check_answer,
    labelled_numbers,
    last_ai_text,
    mismatched_labels,
    turn_observations,
    unsupported_numbers,
    verify_messages,
)

OBS = """
ESA-ADB benchmark metrics (TimeEval ESAScores).
Window: 2000-01-01T00:00:00 → 2000-01-01T08:19:00
Label rows in window: 12  unique event IDs: 3  alarm timesteps: 134
Category filter: Rare Event + Anomaly
  EW_precision=0.681250
  EW_recall=0.400000
  AFF_F_0.50=0.123456
"""


def test_exact_and_rounded_numbers_pass():
    answer = "EW_precision is 0.68 and recall 0.4 over 134 alarm timesteps."
    assert unsupported_numbers(answer, [OBS]) == []


def test_percentage_form_passes():
    """0.6812… stated as 68% is still the tool's number."""
    assert unsupported_numbers("Precision is about 68%.", [OBS]) == []


def test_invented_number_is_flagged():
    answer = "EW_precision is 0.93 with 134 alarm timesteps."
    assert unsupported_numbers(answer, [OBS]) == ["0.93"]


def test_invented_count_is_flagged():
    assert unsupported_numbers("We saw 57 alarm timesteps.", [OBS]) == ["57"]


def test_metric_identifiers_are_not_claims():
    """AFF_F_0.50 / F_0.5 are names, not reported values."""
    answer = "Using AFF_F_0.50 and EW_F_0.50 at beta 0.5, precision is 0.68."
    assert unsupported_numbers(answer, [OBS]) == []


def test_no_tool_output_means_nothing_to_check():
    """Knowledge-only chat with no tool call is not flagged."""
    assert unsupported_numbers("The window size is 250.", []) == []
    assert unsupported_numbers("The window size is 250.", ["   "]) == []


def test_several_flags_are_deduped():
    answer = "Precision 0.93, recall 0.93, F1 0.77."
    assert unsupported_numbers(answer, [OBS]) == ["0.93", "0.77"]


def test_label_mismatch_is_flagged():
    """Recall's value reported as precision — both numbers exist in the tool output."""
    answer = "EW_precision is 0.40 for this slice."
    flags = mismatched_labels(answer, [OBS])
    assert flags == ["0.40 (tool says 0.68125)"]
    # The plain value pass must not report it a second time.
    assert check_answer(answer, [OBS]) == flags


def test_label_match_with_spaces_and_markdown():
    """'**EW precision**: 0.68' is the same label as 'EW_precision=0.681250'."""
    assert mismatched_labels("**EW precision**: 0.68", [OBS]) == []


def test_label_mismatch_in_markdown_table():
    answer = "| Metric | Value |\n|---|---|\n| EW_recall | 0.91 |"
    assert mismatched_labels(answer, [OBS]) == ["0.91 (tool says 0.4)"]


def test_unknown_label_is_left_to_value_check():
    """A label the tool never printed is not a mismatch…"""
    assert mismatched_labels("Mean drift is 0.42", [OBS]) == []
    # …but the number itself still has to come from somewhere.
    assert check_answer("Mean drift is 0.42", [OBS]) == ["0.42"]


def test_settings_are_not_treated_as_metrics():
    """beta/k are inputs, so a different value is not a wrong claim."""
    assert mismatched_labels("beta: 0.5 and k: 3", [OBS]) == []


def test_labelled_numbers_reads_tool_report():
    pairs = labelled_numbers(OBS)
    assert pairs["ewprecision"][0][1] == 0.681250
    assert pairs["ewrecall"][0][1] == 0.400000


class _Msg:
    """Minimal stand-in for LangChain messages (type/content/tool_calls)."""

    def __init__(self, type_: str, content: str, tool_calls=None):
        self.type = type_
        self.content = content
        self.tool_calls = tool_calls or []


def test_turn_observations_only_covers_current_turn():
    messages = [
        _Msg("human", "old question"),
        _Msg("tool", "stale observation 999"),
        _Msg("ai", "old answer"),
        _Msg("human", "new question"),
        _Msg("tool", OBS),
        _Msg("ai", "EW_precision is 0.68."),
    ]
    assert turn_observations(messages) == [OBS]
    assert last_ai_text(messages) == "EW_precision is 0.68."
    assert verify_messages(messages) == []


def test_stale_observation_does_not_support_new_answer():
    messages = [
        _Msg("human", "old question"),
        _Msg("tool", "precision was 0.93"),
        _Msg("ai", "old answer"),
        _Msg("human", "new question"),
        _Msg("tool", OBS),
        _Msg("ai", "EW_precision is 0.93."),
    ]
    # Labelled, so the flag also names what the tool actually printed.
    assert verify_messages(messages) == ["0.93 (tool says 0.68125)"]


def test_last_ai_text_skips_tool_call_turns():
    messages = [
        _Msg("human", "q"),
        _Msg("ai", "", tool_calls=[{"id": "1", "name": "score_telemetry"}]),
        _Msg("tool", OBS),
        _Msg("ai", "Done."),
    ]
    assert last_ai_text(messages) == "Done."
