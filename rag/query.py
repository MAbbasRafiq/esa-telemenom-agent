#!/usr/bin/env python3
"""Query the FAISS index (retrieval only).

Flow:
  question → embed question → FAISS nearest chunks → return / print them

By default this does NOT call an LLM. You only see what RAG retrieves.
The agent later uses retrieve() as a tool.

CLI:

  conda activate esa-agent
  cd esa-telemanom-agent
  python -m rag.query "What is the look-back window size?"
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

# Cached after first load so repeated retrieve() calls stay fast.
_embedder: SentenceTransformer | None = None
_index = None
_chunks: list[dict] | None = None


def _load_index() -> tuple:
    """Load FAISS index, chunk list, and embedder into memory (once)."""
    global _embedder, _index, _chunks

    index_path = INDEX_DIR / "faiss.index"
    if not index_path.exists():
        raise SystemExit(
            "No index yet. From the project root run:\n"
            "  python -m rag.build_index"
        )

    if _index is None:
        _index = faiss.read_index(str(index_path))
        with open(INDEX_DIR / "chunks.pkl", "rb") as f:
            _chunks = pickle.load(f)

    if _embedder is None:
        # Prefer local Hugging Face cache if the model was already downloaded.
        try:
            _embedder = SentenceTransformer(EMBED_MODEL_NAME, local_files_only=True)
        except Exception:
            _embedder = SentenceTransformer(EMBED_MODEL_NAME)

    return _embedder, _index, _chunks


def retrieve(question: str, k: int = 5) -> list[tuple[float, dict]]:
    """Return the top-k (score, chunk) pairs for a natural-language question.

    Each chunk dict has keys: source, id, text.
    Score is cosine similarity (higher is better).
    """
    model, index, chunks = _load_index()

    query_vector = model.encode([question], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vector)

    scores, idxs = index.search(query_vector, k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        results.append((float(score), chunks[idx]))
    return results


def main() -> None:
    """CLI: print top retrieved chunks for a question."""
    parser = argparse.ArgumentParser(description="Retrieve chunks from rag/index/")
    parser.add_argument("question", help="Natural language question")
    parser.add_argument("-k", type=int, default=5, help="How many chunks to retrieve")
    parser.add_argument(
        "--answer",
        action="store_true",
        help="Also print a simple context block stitched from top chunks (no LLM)",
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
        print("CONTEXT FOR ANSWERING (retrieved only — no LLM call):")
        print("=" * 60)
        for score, chunk in hits:
            print(f"[{chunk['source']}] {chunk['text']}\n")


if __name__ == "__main__":
    main()
