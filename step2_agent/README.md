# Step 2 — Single reasoning agent + retrieve tool

Step 1 = retrieve chunks.  
Step 2 = **agent** that calls retrieve, then writes a **short answer**.

```text
Question
  → Reason
  → Tool: retrieve_knowledge  (uses Step 1 FAISS)
  → Observe chunks
  → Short final answer
```

Still **no** Telemanom scoring / HTML. One agent, one tool.

## Setup

Use the **same venv** as Step 1:

```bash
cd esa-telemanom-agent/step1_rag
source .venv/bin/activate
pip install -r ../step2_agent/requirements.txt   # openai + python-dotenv
```

FAISS index must already exist (`python build_index.py` in `step1_rag/`).

### API key via `.env` (recommended)

```bash
cd esa-telemanom-agent
cp .env.example .env
# edit .env → set GROQ_API_KEY=gsk_...
```

`agent.py` loads `../.env` automatically (existing shell `export` still wins).

## Run — offline (no API key)

```bash
cd esa-telemanom-agent/step2_agent
source ../step1_rag/.venv/bin/activate

python agent.py "What is the window size?" --offline
```

## Run — free LLM (recommended: Groq)

**Why Groq:** free API key, very fast, OpenAI-compatible, tool-calling works well for this agent.  
(Gemini also supported; Mistral/Grok not wired by default.)

1. Create a free key: https://console.groq.com/keys  
2. Put it in `.env` (see above) **or** `export GROQ_API_KEY=gsk_...`  
3. Then:

```bash
cd esa-telemanom-agent/step2_agent
source ../step1_rag/.venv/bin/activate

python agent.py "What is the window size?" --llm
python agent.py "What was the lab run status?" --llm
```

If `GROQ_API_KEY` is set (env or `.env`), `--llm` is optional (auto-detected).

### Free alternative: Gemini

```bash
# key from https://aistudio.google.com/apikey — or set GEMINI_API_KEY in .env
export GEMINI_API_KEY=...
python agent.py "What is the window size?" --llm
```

### Provider priority

1. `GROQ_API_KEY` → Groq (`openai/gpt-oss-20b`)  
2. `GEMINI_API_KEY` / `GOOGLE_API_KEY` → Gemini (`gemini-2.0-flash`)  
3. `OPENAI_API_KEY` / `OPENAI_BASE_URL` → OpenAI or local Ollama  

(Values can come from `.env` or shell `export`.)

## What to notice

1. Agent does not read all of `knowledge/` — it **calls a tool**.
2. Answer should stay grounded in retrieved chunks + source names.
3. Next step (later): add a second tool for Telemanom scoring (`model/`).

## Files

| File | Role |
|------|------|
| `agent.py` | Single agent + `retrieve_knowledge` |
| `requirements.txt` | `openai` client (talks to Groq/Gemini/OpenAI) |
