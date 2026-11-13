"""Score + compare regression on a prepared sample (no LLM).

Uses a short max_rows so the suite stays practical on CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import config
import tools


# Short slice for CI/dev machines (still long enough for Telemanom window=250).
SMOKE_MAX_ROWS = 500


@pytest.fixture
def clean_agent_out():
    """Remove previous alarms/meta so each test starts clean."""
    out = config.DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    for name in ("alarms.csv", "scores.csv", "scoring_meta.json"):
        path = out / name
        if path.is_file():
            path.unlink()
    yield out


def test_score_demo_sample_writes_alarms(
    demo_sample_csv: Path,
    model_pack_ok: bool,
    clean_agent_out: Path,
):
    """score_telemetry writes scores, alarms, and scoring_meta for a sample."""
    if not model_pack_ok:
        pytest.skip("model/model.h5 missing")

    msg = tools.tool_score_telemetry(
        str(demo_sample_csv),
        max_rows=SMOKE_MAX_ROWS,
        skip_rows=0,
        threshold=True,
    )
    assert msg.startswith("Scored OK."), msg
    assert (clean_agent_out / "alarms.csv").is_file()
    assert (clean_agent_out / "scores.csv").is_file()
    assert (clean_agent_out / "scoring_meta.json").is_file()

    meta = json.loads((clean_agent_out / "scoring_meta.json").read_text())
    assert meta["csv_path"] == str(demo_sample_csv.resolve())
    assert meta["skip_rows"] == 0
    assert meta["max_rows"] == SMOKE_MAX_ROWS


def test_compare_reuses_existing_score(
    demo_sample_csv: Path,
    model_pack_ok: bool,
    clean_agent_out: Path,
):
    """Second compare on the same slice must not require a fresh score.

    We score once, then call compare. _needs_fresh_score must be False
    so Telemanom does not run twice for the same file/slice.
    """
    if not model_pack_ok:
        pytest.skip("model/model.h5 missing")

    score_msg = tools.tool_score_telemetry(
        str(demo_sample_csv),
        max_rows=SMOKE_MAX_ROWS,
        skip_rows=0,
    )
    assert score_msg.startswith("Scored OK."), score_msg

    alarms_path = clean_agent_out / "alarms.csv"
    meta_path = clean_agent_out / "scoring_meta.json"
    alarms_mtime = alarms_path.stat().st_mtime
    meta_before = meta_path.read_text(encoding="utf-8")

    # Same slice → must not need a new Telemanom run.
    assert tools._needs_fresh_score(demo_sample_csv, 0, SMOKE_MAX_ROWS) is False

    compare_msg = tools.tool_compare_alarms_to_labels(
        str(demo_sample_csv),
        max_rows=SMOKE_MAX_ROWS,
        skip_rows=0,
        re_score=False,
    )
    assert not compare_msg.startswith("Error:"), compare_msg
    # Report should include ESAScores or the fallback note + overlap section.
    assert "Compared alarms" in compare_msg or "ESAScores" in compare_msg or "overlap" in compare_msg.lower()

    # Alarms file should be reused (same mtime), not rewritten by a second score.
    assert alarms_path.stat().st_mtime == alarms_mtime
    assert meta_path.read_text(encoding="utf-8") == meta_before


def test_score_reuses_after_compare(
    demo_sample_csv: Path,
    model_pack_ok: bool,
    clean_agent_out: Path,
):
    """score_telemetry after a scored compare must not re-run Telemanom."""
    if not model_pack_ok:
        pytest.skip("model/model.h5 missing")

    compare_msg = tools.tool_compare_alarms_to_labels(
        str(demo_sample_csv),
        max_rows=SMOKE_MAX_ROWS,
        skip_rows=0,
    )
    assert not compare_msg.startswith("Error:"), compare_msg

    alarms_path = clean_agent_out / "alarms.csv"
    assert alarms_path.is_file()
    alarms_mtime = alarms_path.stat().st_mtime

    score_msg = tools.tool_score_telemetry(
        str(demo_sample_csv),
        max_rows=SMOKE_MAX_ROWS,
        skip_rows=0,
    )
    assert score_msg.startswith("Scored OK"), score_msg
    assert "reused" in score_msg.lower(), score_msg
    assert alarms_path.stat().st_mtime == alarms_mtime
