# ESA Telemanom Agent

Learning workspace: **RAG → single agent → scoring → (later) multi-agent**.

## Layout

```text
esa-telemanom-agent/
  .env.example / .env     ← GROQ_API_KEY (gitignored)
  requirements.txt        ← one env for all steps
  knowledge/             ← RAG corpus
  model/                  ← lab weights + contract
  scoring/                ← thin infer code (no HTML)
  step1_rag/
  step2_agent/
  step3_score/            ← CLI around scoring/ + model/
```

## Steps

| Step | Status | Where |
|------|--------|--------|
| 1. RAG (FAISS) | done | `step1_rag/` |
| 2. Single agent + retrieve | done | `step2_agent/` |
| 3. Thin scoring | done | `scoring/` + `step3_score/` |
| 4. Agent + RAG + score tools | not yet | — |
| 5. Multi-agent | later | — |

## One conda env: `esa-agent`

Use **one** env for this whole folder (not per-step `.venv`).  
Setup commands: see your chat / notes from the agent — activate `esa-agent`, then `pip install -r requirements.txt`.

Prefer **Python 3.9–3.11** (lab `model.h5`). Avoid system Python 3.14 for Step 3.
