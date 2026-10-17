# Step 2 — Single LLM agent + retrieve tool

Step 1 = retrieve chunks.  
Step 2 = **LLM agent** that calls retrieve, then writes a **short answer**.

```text
Question
  → Reason
  → Tool: retrieve_knowledge  (uses Step 1 FAISS)
  → Observe chunks
  → Short final answer
```

Still **no** Telemanom scoring / HTML. One agent, one tool. Always uses the LLM (needs `.env` API key).

## Setup

```bash
conda activate esa-agent
cd esa-telemanom-agent
cp .env.example .env
# edit .env → set GROQ_API_KEY=gsk_...
```

FAISS index must already exist (`python build_index.py` in `step1_rag/`).

## Run

```bash
cd esa-telemanom-agent/step2_agent
conda activate esa-agent

python agent.py "What is the window size?"
python agent.py "What was the lab run status?"
```

### Free alternative: Gemini

```bash
# key from https://aistudio.google.com/apikey — or set GEMINI_API_KEY in .env
python agent.py "What is the window size?"
```

### Provider priority

1. `GROQ_API_KEY` → Groq (`openai/gpt-oss-20b`)  
2. `GEMINI_API_KEY` / `GOOGLE_API_KEY` → Gemini (`gemini-2.0-flash`)  
3. `OPENAI_API_KEY` / `OPENAI_BASE_URL` → OpenAI or local Ollama  

## What to notice

1. Agent does not read all of `knowledge/` — it **calls a tool**.
2. Answer should stay grounded in retrieved chunks + source names.
3. Next step: Step 4 adds score + compare tools (+ Streamlit UI).

## Files

| File | Role |
|------|------|
| `agent.py` | Single LLM agent + `retrieve_knowledge` |
| `requirements.txt` | `openai` client (talks to Groq/Gemini/OpenAI) |
