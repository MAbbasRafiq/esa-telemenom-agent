# Step 1 — RAG only (FAISS)

Goal: understand **retrieval**. No agent. No Telemanom scoring. No HTML reports.

```text
knowledge/ files
    → build_index.py   (chunk → embed → FAISS)
    → query.py         (question → similar chunks)
```

## Setup (once)

```bash
cd esa-telemanom-agent/step1_rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python build_index.py
```

First `build_index.py` download karega embedding model (`all-MiniLM-L6-v2`) — internet chahiye.

## Query

```bash
source .venv/bin/activate
python query.py "Which channels are Target YES in subsystem_6?"
python query.py "What was the lab run status and dataset?" -k 3
python query.py "What is the LSTM window size?" --answer
```

Output = top matching **chunks** + similarity score.  
Yahi RAG ka “R” (Retrieval) hai.

## What you should notice

1. Sawal ka answer invent nahi ho raha — pehle documents se pieces aa rahe hain.
2. Galat / vague sawal pe irrelevant chunks bhi aa sakte hain — isliye chunking matter karti hai.
3. LLM agent abhi nahi: tum khud chunks padh ke jawab banao. Step 2 mein agent yahi retrieve tool use karega.

## Files

| File | Role |
|------|------|
| `build_index.py` | knowledge → FAISS |
| `query.py` | question → top chunks |
| `index/` | generated (gitignored) |

## Not in Step 1

- Elasticsearch (baad mein optional)
- Reasoning agent
- `model/model.h5` inference
