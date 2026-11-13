# Tool contracts (single-agent)

Stable input/output rules for the three tools in `agent/tools.py`.  
Keep these fixed before adding multi-agent routing.

All tools return a **plain string** (for the LLM observation).  
Failures always start with `Error:` so callers can detect them easily.

---

## 1. `retrieve_knowledge`

| | |
|--|--|
| **Purpose** | Search FAISS over `knowledge/` |
| **Inputs** | `query` (str), `k` (int, default 5, clamped to 1–8) |
| **Success** | Ranked text blocks with score, source, id |
| **Empty** | `"No chunks found."` |
| **Does not** | Call Telemanom or write files |

---

## 2. `score_telemetry`

| | |
|--|--|
| **Purpose** | Run Telemanom on a telemetry CSV slice; short alarm-channel summary |
| **Inputs** | `csv_path` (required), `max_rows` (default 3000), `skip_rows` (default 0), `threshold` (default true) |
| **Success** | Starts with `Scored OK.` — writes `agent/out/scores.csv`, `alarms.csv`, and `scoring_meta.json` |
| **Reuse** | If `scoring_meta.json` already matches this CSV/slice and `alarms.csv` exists → **reuse** (no Telemanom). Message includes `Scored OK (reused previous run…)` |
| **Failure** | Starts with `Error:` (missing CSV, missing model, runtime error) |
| **Side effects** | On a fresh run, overwrites previous `agent/out/` artifacts for this project |

---

## 3. `compare_alarms_to_labels`

| | |
|--|--|
| **Purpose** | Compare alarms to ground-truth labels (+ ESAScores) |
| **Inputs** | `csv_path` (required), `max_rows`, `skip_rows`, `re_score` (default false) |
| **Success** | Text report with ESAScores + simple interval overlap |
| **Scoring rule** | Scores only when alarms are missing or the CSV/slice differs from `scoring_meta.json`. Same slice already scored → **reuse** (no second Telemanom pass). |
| **Empty slice** | No labelled event in the scored window → report says ESAScores were **not computed** and precision/recall are undefined (interval overlap still shown). Never a bare `AssertionError`. |
| **Failure** | Starts with `Error:` |

**Agent usage rule:** call **one** of `score_telemetry` or `compare_alarms_to_labels` per turn — never both.  
For “score and compare”, call `compare_alarms_to_labels` once.

---

## Dispatch

`dispatch_tool(name, args, default_k)` is the only switch the LLM loop uses.  
Unknown `name` → `"Unknown tool: …"`.

Tool names in `TOOLS_SPEC` must match the names handled in `dispatch_tool`.
