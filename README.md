# ESA Telemanom Agent

Single LLM agent over **RAG** + **Telemanom scoring** + **label compare**.  
CLI only (no UI yet). One conda env: `esa-agent`.

## Layout

```text
esa-telemanom-agent/
  knowledge/     ← RAG corpus
  model/         ← lab weights + contract
  rag/           ← FAISS build + retrieve
  scoring/       ← Telemanom infer + ESAScores
  agent/         ← CLI agent (3 tools)
    config.py    ← paths + .env
    tools.py     ← retrieve / score / compare
    llm.py       ← provider + tool-calling loop
    agent.py     ← terminal entrypoint
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

## Tools

| Tool | Uses |
|------|------|
| `retrieve_knowledge` | FAISS + `knowledge/` |
| `score_telemetry` | `model/` + `scoring/` |
| `compare_alarms_to_labels` | alarms vs `is_anomaly_*` (+ ESAScores) |

No LangChain / LangGraph — custom tool-calling loop in `agent/llm.py`.
