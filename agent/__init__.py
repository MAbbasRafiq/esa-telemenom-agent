"""CLI agent package: one LLM with three tools (retrieve, score, compare).

Modules:
  config.py  — paths, .env, sys.path
  tools.py   — retrieve / score / compare + TOOLS_SPEC
  llm.py     — provider config + tool-calling loop
  agent.py   — terminal CLI entrypoint

Run:

  conda activate esa-agent
  cd esa-telemanom-agent/agent
  python agent.py "What is the window size?"
"""
