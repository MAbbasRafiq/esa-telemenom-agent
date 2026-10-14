#!/usr/bin/env python3
"""Step 1 — Build a FAISS index from ../knowledge/

Learning flow (this script = first half of RAG):
  1) Load knowledge files
  2) Split into small text "chunks"
  3) Embed each chunk (sentence-transformers → vectors)
  4) Store vectors in FAISS + save chunk texts beside the index

No LLM answering here. No Telemanom model. Retrieval only setup.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "knowledge"
INDEX_DIR = Path(__file__).resolve().parent / "index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def chunks_from_channels(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    out = []
    for _, row in df.iterrows():
        text = (
            f"ESA Mission1 channel metadata. "
            f"Channel: {row['Channel']}. "
            f"Subsystem: {row['Subsystem']}. "
            f"Physical unit: {row['Physical Unit']}. "
            f"Group: {row['Group']}. "
            f"Target channel for anomaly detection: {row['Target']}. "
            f"Categorical: {row['Categorical']}."
        )
        out.append({"source": path.name, "id": str(row["Channel"]), "text": text})
    return out


def chunks_from_anomaly_types(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    out = []
    for _, row in df.iterrows():
        text = (
            f"ESA Mission1 anomaly type. "
            f"ID: {row['ID']}. Class: {row['Class']}. Subclass: {row['Subclass']}. "
            f"Category: {row['Category']}. Dimensionality: {row['Dimensionality']}. "
            f"Locality: {row['Locality']}. Length: {row['Length']}."
        )
        out.append({"source": path.name, "id": str(row["ID"]), "text": text})
    return out


def chunks_from_lab_results(path: Path) -> list[dict]:
    """Wide results.csv → a few readable summary chunks (skip per-channel columns)."""
    df = pd.read_csv(path)
    row = df.iloc[0]
    skip = ("channel_",)
    parts = []
    for col in df.columns:
        if any(s in col for s in skip):
            continue
        val = row[col]
        if pd.isna(val):
            continue
        parts.append(f"{col}: {val}")

    # One overview chunk + smaller metric-group chunks for better retrieval
    overview = (
        "Lab Telemanom-ESA full run summary. "
        + " | ".join(parts[:12])
    )
    metrics = (
        "Lab Telemanom-ESA evaluation metrics (non-channel). "
        + " | ".join(parts[12:])
    )
    return [
        {"source": path.name, "id": "lab_overview", "text": overview},
        {"source": path.name, "id": "lab_metrics", "text": metrics},
    ]


def chunks_from_text(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    # Small file: keep as one chunk; if long, split on blank lines
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if not blocks:
        blocks = [text]
    out = []
    for i, block in enumerate(blocks):
        out.append({"source": path.name, "id": f"{path.stem}_{i}", "text": block})
    return out


def load_all_chunks() -> list[dict]:
    chunks: list[dict] = []
    chunks += chunks_from_channels(KNOWLEDGE / "channels.csv")
    chunks += chunks_from_anomaly_types(KNOWLEDGE / "anomaly_types.csv")
    chunks += chunks_from_lab_results(KNOWLEDGE / "lab_results_84months.csv")
    chunks += chunks_from_text(KNOWLEDGE / "model_card.txt")
    return chunks


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_all_chunks()
    print(f"Loaded {len(chunks)} chunks from {KNOWLEDGE}")

    print(f"Embedding with {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)  # cosine similarity via inner product

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    meta = {
        "embed_model": EMBED_MODEL,
        "n_chunks": len(chunks),
        "dim": int(vectors.shape[1]),
        "knowledge_dir": str(KNOWLEDGE),
    }
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote FAISS index → {INDEX_DIR}")
    print("Next: python query.py \"Which channels are Target YES?\"")


if __name__ == "__main__":
    main()
