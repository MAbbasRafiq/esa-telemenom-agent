#!/usr/bin/env python3
"""Step 1 — Query the FAISS index (retrieval).

Learning flow (this script = second half of RAG retrieval):
  question → embed question → FAISS nearest chunks → print them

By default this does NOT call an LLM. You see what RAG *retrieves*.
That is the important part to understand before adding an agent.

Optional: --answer uses a very simple extractive join of top chunks
(still no external LLM API) so you can read a combined context block.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).resolve().parent / "index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_embedder: SentenceTransformer | None = None
_index = None
_chunks: list[dict] | None = None


def _load_index() -> tuple:
    global _embedder, _index, _chunks
    if not (INDEX_DIR / "faiss.index").exists():
        raise SystemExit(
            "No index yet. From step1_rag/ run: python build_index.py"
        )
    if _index is None:
        _index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        with open(INDEX_DIR / "chunks.pkl", "rb") as f:
            _chunks = pickle.load(f)
    if _embedder is None:
        # Prefer local HF cache (already downloaded during Step 1) — no Hub call.
        try:
            _embedder = SentenceTransformer(EMBED_MODEL_NAME, local_files_only=True)
        except Exception:
            _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder, _index, _chunks


def retrieve(question: str, k: int = 5) -> list[tuple[float, dict]]:
    """Return top-k (score, chunk) from the Step 1 FAISS index."""
    model, index, chunks = _load_index()
    q = model.encode([question], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q)

    scores, idxs = index.search(q, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        results.append((float(score), chunks[idx]))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1 RAG retrieve from knowledge/")
    parser.add_argument("question", help="Natural language question")
    parser.add_argument("-k", type=int, default=5, help="How many chunks to retrieve")
    parser.add_argument(
        "--answer",
        action="store_true",
        help="Also print a simple context block stitched from top chunks (no LLM API)",
    )
    args = parser.parse_args()

    hits = retrieve(args.question, k=args.k)
    print(f"\nQuestion: {args.question}\n")
    print(f"Top {len(hits)} chunks:\n")
    for rank, (score, chunk) in enumerate(hits, 1):
        print(f"--- {rank}  score={score:.3f}  source={chunk['source']}  id={chunk['id']}")
        print(chunk["text"])
        print()

    if args.answer:
        print("=" * 60)
        print("CONTEXT FOR ANSWERING (retrieved only — you / a future LLM use this):")
        print("=" * 60)
        for score, chunk in hits:
            print(f"[{chunk['source']}] {chunk['text']}\n")


if __name__ == "__main__":
    main()
