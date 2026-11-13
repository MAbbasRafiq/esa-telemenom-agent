"""Empty-slice guard for ESAScores.

ESAScores asserts internally when no ground-truth event overlaps the scored
window, which used to surface as a bare `AssertionError:` in the answer.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scoring.esa_metrics import (
    DEFAULT_ANOMALY_TYPES,
    DEFAULT_LABELS_CSV,
    NoLabeledEventsInWindow,
    compute_esa_scores,
)


def _alarms_frame(channel: str) -> pd.DataFrame:
    index = pd.date_range("2000-01-01", periods=20, freq="30s")
    return pd.DataFrame({channel: [0] * 20}, index=index)


def test_no_labelled_event_raises_typed_error():
    if not Path(DEFAULT_LABELS_CSV).is_file():
        pytest.skip(f"labels.csv not available: {DEFAULT_LABELS_CSV}")
    if not Path(DEFAULT_ANOMALY_TYPES).is_file():
        pytest.skip(f"anomaly_types.csv not available: {DEFAULT_ANOMALY_TYPES}")

    # Channel that appears in no label row → empty y_true for any window.
    alarms = _alarms_frame("channel_does_not_exist")

    with pytest.raises(NoLabeledEventsInWindow) as excinfo:
        compute_esa_scores(alarms)

    message = str(excinfo.value)
    assert "No ground-truth event" in message
    # The message must name the window so the analyst can widen the slice.
    assert "2000-01-01T00:00:00" in message


def test_compare_tool_explains_empty_slice(monkeypatch, tmp_path):
    """The tool turns the typed error into guidance, not a raw traceback."""
    import tools

    alarms = _alarms_frame("channel_41")
    alarms_path = tmp_path / "alarms.csv"
    alarms.to_csv(alarms_path)

    csv_path = tmp_path / "slice.csv"
    pd.DataFrame(
        {
            "timestamp": alarms.index,
            "channel_41": [0.0] * len(alarms),
            "is_anomaly_channel_41": [0] * len(alarms),
        }
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(tools, "_needs_fresh_score", lambda *a, **k: False)

    out = tools.tool_compare_alarms_to_labels(
        csv_path=str(csv_path),
        max_rows=len(alarms),
        skip_rows=0,
        alarms_csv=str(alarms_path),
    )

    assert "no labelled event" in out.lower()
    assert "max_rows" in out
    assert "Traceback" not in out
