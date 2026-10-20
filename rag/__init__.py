"""RAG package: build a FAISS index over knowledge/ and retrieve chunks.

Public helpers:
  - build_index.py  → run once to create rag/index/
  - query.retrieve  → used by the agent tool retrieve_knowledge
"""

from .query import retrieve

__all__ = ["retrieve"]
