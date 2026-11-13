"""Per-turn run log (no LLM, no Telemanom)."""
from __future__ import annotations

import json

import runlog


def test_log_turn_appends_one_json_line(tmp_path):
    path = tmp_path / "runs.jsonl"

    runlog.log_turn(
        thread_id="t1",
        question="Score this telemetry",
        route="score",
        tools=["score_telemetry"],
        duration_ms=1234,
        provider="groq",
        model="openai/gpt-oss-20b",
        max_rows=500,
        reused_alarms=True,
        unverified=["0.93"],
        answer_chars=120,
        path=path,
    )

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["thread_id"] == "t1"
    assert record["route"] == "score"
    assert record["tools"] == ["score_telemetry"]
    assert record["duration_ms"] == 1234
    assert record["reused_alarms"] is True
    assert record["unverified_count"] == 1
    assert record["error"] is None
    assert record["ts"]


def test_long_questions_are_truncated(tmp_path):
    path = tmp_path / "runs.jsonl"
    runlog.log_turn(thread_id="t1", question="x" * 1000, path=path)

    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert len(record["question"]) == runlog.MAX_QUESTION_CHARS


def test_read_turns_is_newest_first(tmp_path):
    path = tmp_path / "runs.jsonl"
    for i in range(3):
        runlog.log_turn(thread_id=f"t{i}", question="q", path=path)

    records = runlog.read_turns(limit=2, path=path)
    assert [r["thread_id"] for r in records] == ["t2", "t1"]


def test_log_never_raises_on_bad_target(tmp_path):
    """Logging must not break a run, whatever the filesystem does."""
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")

    runlog.log_turn(thread_id="t1", question="q", path=blocker / "runs.jsonl")
    assert runlog.read_turns(path=blocker / "runs.jsonl") == []


def test_rotation_keeps_one_old_file(tmp_path, monkeypatch):
    path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(runlog, "RUNLOG_MAX_BYTES", 200)

    for i in range(6):
        runlog.log_turn(thread_id=f"t{i}", question="q" * 50, path=path)

    assert path.is_file()
    assert (tmp_path / "runs.jsonl.1").is_file()
