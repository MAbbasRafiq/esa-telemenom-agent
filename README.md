# ESA Telemanom Agent

Single LLM agent over **RAG** + **Telemanom scoring** + **label compare**.  
CLI + **FastAPI** + **React** UI. One conda env: `esa-agent` (Python).  
Frontend needs Node.js (also installable into `esa-agent` via conda-forge).

## Layout

```text
esa-telemanom-agent/
  knowledge/     ← RAG corpus
  model/         ← lab weights + contract
  rag/           ← FAISS build + retrieve
  scoring/       ← Telemanom infer + ESAScores
  agent/         ← CLI + LangGraph multi-agent (3 tools)
    config.py    ← paths + .env
    tools.py     ← retrieve / score / compare (contracts)
    lc_tools.py  ← LangChain wrappers + observation trim
    session.py   ← per-thread CSV / defaults / chat titles
    multi_agent.py ← Router → Knowledge | Score | Compare → Verifier
    verify.py    ← numeric grounding check (no LLM)
    runlog.py    ← one JSON line per turn (data/runs.jsonl)
    graph.py     ← compiled graph + SQLite checkpointer
    llm.py       ← provider config + run_agent() entry
    agent.py     ← terminal entrypoint
  api/           ← FastAPI HTTP wrapper (thread_id sessions)
  frontend/      ← React + Vite + Tailwind
  samples/       ← small CSV windows for demos / tests
  docs/          ← TOOL_CONTRACTS.md (stable tool I/O)
  tests/         ← pytest regression (RAG + score/compare + routing)
  data/          ← checkpoints, sessions.json, runs.jsonl (gitignored)
```

## Setup

```bash
conda activate esa-agent
cd ~/ESA_ADB/esa-telemanom-agent
pip install -r requirements.txt
cp .env.example .env   # set GROQ_API_KEY=...
```

Prefer Python 3.9–3.11 so TensorFlow can load lab `model.h5`.

Build the FAISS index once (or after changing `knowledge/`):

```bash
python -m rag.build_index
```

## RAG only (no LLM)

```bash
python -m rag.query "What is the look-back window size?"
```

## Agent (CLI)

```bash
cd agent

# Facts from knowledge/
python agent.py "What is the window size?"

# Score + compare to ground truth
python agent.py "Score the slice and compare alarms to ground-truth labels. Summarize." \
  --csv ../../data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.test.csv \
  --skip-rows 33064 --max-rows 3000
```

Outputs (scores / alarms) go to `agent/out/`.

## FastAPI (HTTP)

From the **project root** (not `agent/`):

```bash
conda activate esa-agent
cd ~/ESA_ADB/esa-telemanom-agent
uvicorn api.main:app --reload --host 127.0.0.1 --port 8100
```

Ports for **this** project (so it can sit beside another app on 8000 / 5173):

| Service | Port |
|---------|------|
| FastAPI | **8100** |
| React (Vite) | **5180** |

- Docs: http://127.0.0.1:8100/docs  
- Health: http://127.0.0.1:8100/health  

Example:

```bash
curl -X POST http://127.0.0.1:8100/agent/run \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the look-back window size?", "k": 3}'
```

CORS allows the React origin on port 5180.

## React UI

Needs **two terminals**: API + Vite.

```bash
# Terminal 1 — FastAPI
conda activate esa-agent
cd ~/ESA_ADB/esa-telemanom-agent
uvicorn api.main:app --reload --host 127.0.0.1 --port 8100

# Terminal 2 — React (Node required)
conda activate esa-agent   # if node was installed into this env
cd ~/ESA_ADB/esa-telemanom-agent/frontend
npm install
npm run dev
```

Open http://127.0.0.1:5180 — the UI queues work on `POST /agent/jobs`.

Analyst UI: question + **CSV upload** (or prepared samples under `samples/`).  
No ports, paths, or `skip_rows` on screen. Original `84_months.test.csv` is untouched.

The left rail lists past **Chats** (`GET /sessions`); clicking one replays it from
the checkpointer and keeps writing to that same `thread_id`, and `×` forgets it
(`DELETE /sessions/{id}`). Answers show the route badge and, when the verifier
flags something, an "unverified figures" note.

Optional: `frontend/.env` with `VITE_API_BASE` if the API is not on port 8100.

## Tools

| Tool | Uses | Owned by |
|------|------|----------|
| `retrieve_knowledge` | FAISS + `knowledge/` | `knowledge` agent |
| `score_telemetry` | `model/` + `scoring/` | `score` agent |
| `compare_alarms_to_labels` | alarms vs `is_anomaly_*` (+ ESAScores) | `compare` agent |

Each specialist holds exactly one tool, so a single turn can never run Telemanom
twice. Contracts: `docs/TOOL_CONTRACTS.md`.

**Runtime:** LangGraph multi-agent (`agent/graph.py`) with **SQLite** checkpointing
under `data/langgraph_checkpoints.sqlite`, plus CSV hints and chat titles in
`data/sessions.json`. Pass / reuse `thread_id` so follow-ups keep memory across
API restarts (same machine).

**Sessions:** `GET /sessions` lists past chats newest first,
`GET /sessions/{id}` replays one as question/answer turns (read straight from the
checkpointer, so it works without an API key), `DELETE /sessions/{id}` drops the
session and its history. A chat is titled by its first question.

**Async jobs (UI):** `POST /agent/jobs` queues a run; poll `GET /agent/jobs/{id}`.
Scoring no longer blocks the HTTP request. Sync `POST /agent/run-upload` remains
for curl. Set `ESA_AGENT_LEGACY=1` for the old DIY loop (no thread history).

**Real-data guards:** UI uploads capped at ~50 MB / 20k rows (env:
`ESA_MAX_UPLOAD_BYTES`, `ESA_MAX_UI_SCORE_ROWS`). Tool `max_rows` clamped to 50k
(`ESA_MAX_TOOL_SCORE_ROWS`). Finished jobs purged after ~1h / max 100 kept.

**Multi-agent:** `Router` picks one specialist per turn —
`knowledge` (RAG), `score` (`score_telemetry` only), or `compare`
(`compare_alarms_to_labels` only). Response includes `routed_to`, shown as a
badge in the UI. Tool calls pin `csv_path` / `max_rows` / `skip_rows` to the
thread session, so a later compare reuses the alarms a prior score wrote.
Routing is covered by `tests/test_routing.py` (no LLM needed).

**Answer check:** a `verifier` node runs after every specialist
(`agent/verify.py`, pure Python — no extra LLM call), in two passes:

1. **Label check** — `EW_precision = 0.40` in the answer must equal what the tool
   printed for that name, so recall reported as precision is caught
   (`0.40 (tool says 0.68125)`).
2. **Value check** — any remaining number must appear somewhere in this turn's
   tool output.

Flags come back as `unverified_claims` and the UI marks them under the answer.
Rounding (`0.681250` → `0.68`), percentages (`68%`) and metric names
(`AFF_F_0.50`) are accepted; a stale observation from an earlier turn is not.

**Run log:** every turn appends one JSON line to `data/runs.jsonl` — route, tools,
`duration_ms`, `max_rows`, whether alarms were reused, unverified count, and any
error. Rotated at 5 MB (`ESA_RUNLOG_MAX_BYTES`); read it with
`runlog.read_turns()`. This is how "why was that slow / wrong" gets answered
without guessing.

**Empty slices:** if no labelled event overlaps the scored window,
`compute_esa_scores` raises `NoLabeledEventsInWindow` and `compare` explains
that precision/recall are undefined for that slice instead of returning a bare
`AssertionError`.

## Tests (regression)

Score/compare tests need FAISS index + `model/model.h5` +
`samples/demo_known_events.csv`. Routing, verifier, run-log and session tests are
pure Python. Nothing here calls the LLM.

| File | Covers |
|------|--------|
| `test_retrieve.py` | FAISS retrieval |
| `test_tool_contracts.py` | tool I/O shapes |
| `test_score_compare.py` | scoring + alarms reuse |
| `test_routing.py` | router keywords + one-tool-per-specialist |
| `test_verify.py` | label mismatches + unsupported numbers |
| `test_esa_metrics_guard.py` | empty-slice message |
| `test_runlog.py` | `data/runs.jsonl` lines + rotation |
| `test_sessions.py` | chat titles, ordering, history replay |

```bash
conda activate esa-agent
cd ~/ESA_ADB/esa-telemanom-agent
pip install -r requirements.txt
python -m rag.build_index   # if index missing
pytest tests/ -q
```

Score/compare tests use a short `max_rows=500` slice so CPU runs stay reasonable.
