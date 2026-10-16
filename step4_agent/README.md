# Step 4 — Agent: RAG + score + label compare

One reasoning agent, **three tools**:

| Tool | Uses | When |
|------|------|------|
| `retrieve_knowledge` | FAISS + `knowledge/` | facts / config |
| `score_telemetry` | `model/` + `scoring/` | run Telemanom → alarm summary |
| `compare_alarms_to_labels` | alarms vs `is_anomaly_*` in CSV | true / wrong / miss vs ground truth |

`k` for retrieve is hard-capped at **8**.

## Run

```bash
conda activate esa-agent
cd esa-telemanom-agent/step4_agent
```

### RAG only
```bash
python agent.py "What is the look-back window size?" --llm
```

### Score
```bash
python agent.py "Score this slice; how many channels alarmed and top 5?" --llm \
  --csv ../../data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.test.csv \
  --skip-rows 31000 --max-rows 2000
```

### Score + compare to ground truth (new)
```bash
python agent.py "Score the slice and compare alarms to ground-truth labels. Summarize true vs wrong vs miss." --llm \
  --csv ../../data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.test.csv \
  --skip-rows 31000 --max-rows 2000
```

Reuse existing `out/alarms.csv` without re-scoring (ask the model / offline with `re_score=false` via tool args). Offline demo:

```bash
python agent.py "compare" --offline \
  --csv ../../data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.test.csv \
  --skip-rows 31000 --max-rows 2000
```

## Label compare meaning

On each **target channel** in this slice:

| Term | Meaning |
|------|---------|
| **true** | Alarm interval overlaps a labelled anomaly interval |
| **wrong** | Alarm interval with no label overlap |
| **miss** | Labelled interval with no alarm overlap |

Only channels that have ≥1 label in the slice count toward precision/recall.  
Alarms on channels with **no** label in this window are reported separately (not counted as wrong).

Logic: `scoring/eval_labels.py` (same idea as the old HTML report, text-only).
