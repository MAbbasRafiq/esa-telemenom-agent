# ESA Telemanom Agent

Learning workspace: **RAG → agent → scoring → label compare → (later) multi-agent**.

## Layout

```text
esa-telemanom-agent/
  knowledge/              ← RAG corpus
  model/                  ← lab weights + contract
  scoring/                ← infer + eval_labels (no HTML)
  step1_rag/
  step2_agent/            ← RAG-only agent
  step3_score/            ← scoring CLI
  step4_agent/            ← retrieve + score + compare_to_labels
```

## Steps

| Step | Status | Where |
|------|--------|--------|
| 1. RAG (FAISS) | done | `step1_rag/` |
| 2. Single agent + retrieve | done | `step2_agent/` |
| 3. Thin scoring | done | `scoring/` + `step3_score/` |
| 4. Agent + RAG + score + **GT compare** + **Streamlit UI** | **done** | `step4_agent/` (`agent.py`, `app.py`) |
| 5. Multi-agent | later | — |

## One conda env: `esa-agent`

```bash
conda activate esa-agent
cd ~/ESA_ADB/esa-telemanom-agent
pip install -r requirements.txt   # includes streamlit
```

Prefer Python 3.9–3.11 for TensorFlow + lab `model.h5`.

## Streamlit UI

```bash
conda activate esa-agent
cd ~/ESA_ADB/esa-telemanom-agent/step4_agent
streamlit run app.py
```

Single agent, 3 tools, no LangChain/LangGraph.
