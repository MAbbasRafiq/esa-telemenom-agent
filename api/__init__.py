"""HTTP API for the ESA Telemanom agent (FastAPI).

This package does NOT re-implement RAG or Telemanom.
It only exposes the existing agent (agent/llm.py) over HTTP
so a React frontend (or curl) can call it later.
"""
