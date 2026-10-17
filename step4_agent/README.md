# Step 4 — LLM agent: RAG + score + label compare (+ Streamlit UI)

One reasoning agent, **three tools** (always calls the LLM — needs `.env` API key):

| Tool | Uses | When |
|------|------|------|
| `retrieve_knowledge` | FAISS + `knowledge/` | facts / config |
| `score_telemetry` | `model/` + `scoring/` | run Telemanom → alarm summary |
| `compare_alarms_to_labels` | alarms vs `is_anomaly_*` in CSV | true / wrong / miss vs ground truth |

`k` for retrieve is hard-capped at **8**.

No LangChain / LangGraph — custom tool-calling loop in `agent.py`.

## Streamlit UI

```bash
conda activate esa-agent
cd ~/ESA_ADB/esa-telemanom-agent/step4_agent
streamlit run app.py
```

Sidebar: CSV path, `skip_rows` / `max_rows`, RAG `k`.  
Main page: question + **Run agent** → answer + tool steps.

Demo defaults: `skip_rows=295000`, `max_rows=3500` (slice where Telemanom often overlaps GT labels).

**Recall note:** `true` counts *alarm* intervals that hit a label. Several alarms can hit one label, so `true/labels` can exceed 1. We report **event recall = labels_hit / labels** (`labels_hit = label_events − miss`), which is always ≤ 1.

## CLI (same agent)

```bash
conda activate esa-agent
cd esa-telemanom-agent/step4_agent
```

### RAG only
```bash
python agent.py "What is the look-back window size?"
```

### Score + compare to ground truth
```bash
python agent.py "Score the slice and compare alarms to ground-truth labels. Summarize true vs wrong vs miss." \
  --csv ../../data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.test.csv \
  --skip-rows 33064 --max-rows 3000
```

## Metrics (compare tool)

**Primary — ESA-ADB `ESAScores`** (`scoring/esa_metrics.py`), same as lab TimeEval:

- `alarming_precision`
- `EW_precision`, `EW_recall`, `EW_F_0.50`
- `AFF_precision`, `AFF_recall`, `AFF_F_0.50`

Needs `../data/ESA-Mission1/labels.csv` + `anomaly_types.csv`, and `pip install portion`.

**Secondary — intuition only:** true / wrong / miss interval overlap (`scoring/eval_labels.py`).
