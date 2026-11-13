"""Chat sessions: titles, ordering, and history replay (no LLM)."""
from __future__ import annotations

from session import MAX_TITLE_CHARS, SessionStore


def _store(tmp_path) -> SessionStore:
    return SessionStore(path=tmp_path / "sessions.json")


def test_first_question_becomes_the_title(tmp_path):
    store = _store(tmp_path)
    store.upsert("t1", title="Which channels show alarms?")
    store.upsert("t1", title="And the window size?")

    # The sidebar label must not change under the user mid-chat.
    assert store.get("t1").title == "Which channels show alarms?"


def test_long_titles_are_truncated(tmp_path):
    store = _store(tmp_path)
    store.upsert("t1", title="x" * 300)
    assert len(store.get("t1").title) == MAX_TITLE_CHARS


def test_sessions_are_listed_newest_first(tmp_path):
    store = _store(tmp_path)
    store.upsert("old", title="old chat")
    store.upsert("new", title="new chat")
    store.upsert("old", title="ignored")  # touches updated_at

    assert [s.thread_id for s in store.list_sessions()] == ["old", "new"]


def test_titles_survive_a_restart(tmp_path):
    path = tmp_path / "sessions.json"
    first = SessionStore(path=path)
    first.upsert("t1", title="Compare alarms to labels", csv_path="/tmp/a.csv")

    reloaded = SessionStore(path=path)
    session = reloaded.get("t1")
    assert session.title == "Compare alarms to labels"
    assert session.csv_path == "/tmp/a.csv"
    assert session.created_at


def test_reset_forgets_the_session(tmp_path):
    store = _store(tmp_path)
    store.upsert("t1", title="chat")
    store.reset("t1")
    assert store.list_sessions() == []


class _Msg:
    def __init__(self, type_: str, content: str, name: str = "", tool_calls=None):
        self.type = type_
        self.content = content
        self.name = name
        self.tool_calls = tool_calls or []


def test_history_is_grouped_into_turns():
    from graph import _messages_to_turns

    messages = [
        _Msg("human", "Which channels show alarms?"),
        _Msg("ai", "", tool_calls=[{"id": "1", "name": "score_telemetry"}]),
        _Msg("tool", "Scored OK.", name="score_telemetry"),
        _Msg("ai", "channel_12 and channel_35 alarm."),
        _Msg("human", "What is the window size?"),
        _Msg("tool", "window_size=250", name="retrieve_knowledge"),
        _Msg("ai", "250 timesteps."),
    ]

    turns = _messages_to_turns(messages)
    assert [t["question"] for t in turns] == [
        "Which channels show alarms?",
        "What is the window size?",
    ]
    assert turns[0]["answer"] == "channel_12 and channel_35 alarm."
    assert turns[0]["tools"] == ["score_telemetry"]
    assert turns[1]["tools"] == ["retrieve_knowledge"]


def test_unanswered_turn_is_kept_with_empty_answer():
    from graph import _messages_to_turns

    turns = _messages_to_turns([_Msg("human", "pending question")])
    assert turns == [{"question": "pending question", "answer": "", "tools": []}]
