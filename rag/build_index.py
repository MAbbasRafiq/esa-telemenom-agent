#!/usr/bin/env python3
"""Build a FAISS index from the knowledge/ folder.

What this script does (in order):
  1. Load text from knowledge/ (channels, anomaly types, model card, lab summary)
  2. Split that text into small chunks
  3. Embed each chunk with sentence-transformers (vectors)
  4. Store the vectors in FAISS and save chunk texts next to the index

This does NOT call an LLM and does NOT run Telemanom.
Run once (or again after you change knowledge/ files):

  conda activate esa-agent
  cd esa-telemanom-agent
  python -m rag.build_index
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Project root = parent of this rag/ folder
ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "knowledge"
INDEX_DIR = Path(__file__).resolve().parent / "index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def chunks_from_channels(path: Path) -> list[dict]:
    """Turn each row of channels.csv into one searchable text chunk."""
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
    """Turn each row of anomaly_types.csv into one searchable text chunk."""
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
    """Turn the wide lab results CSV into a few readable summary chunks.

    We skip per-channel columns (names starting with channel_) so the
    index stays small and retrieval focuses on overall metrics.
    """
    df = pd.read_csv(path)
    row = df.iloc[0]
    parts = []
    for col in df.columns:
        if col.startswith("channel_"):
            continue
        val = row[col]
        if pd.isna(val):
            continue
        parts.append(f"{col}: {val}")

    # First chunk: overview fields. Second chunk: remaining metrics.
    overview = "Lab Telemanom-ESA full run summary. " + " | ".join(parts[:12])
    metrics = "Lab Telemanom-ESA evaluation metrics (non-channel). " + " | ".join(parts[12:])
    return [
        {"source": path.name, "id": "lab_overview", "text": overview},
        {"source": path.name, "id": "lab_metrics", "text": metrics},
    ]


def chunks_from_text(path: Path) -> list[dict]:
    """Split a plain-text file into chunks on blank lines."""
    text = path.read_text(encoding="utf-8").strip()
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    if not blocks:
        blocks = [text]

    out = []
    for i, block in enumerate(blocks):
        out.append({"source": path.name, "id": f"{path.stem}_{i}", "text": block})
    return out


def load_all_chunks() -> list[dict]:
    """Load every knowledge file and return a flat list of chunks."""
    chunks: list[dict] = []
    chunks += chunks_from_channels(KNOWLEDGE / "channels.csv")
    chunks += chunks_from_anomaly_types(KNOWLEDGE / "anomaly_types.csv")
    chunks += chunks_from_lab_results(KNOWLEDGE / "lab_results_84months.csv")
    chunks += chunks_from_text(KNOWLEDGE / "model_card.txt")
    return chunks


def main() -> None:
    """Embed all chunks and write FAISS index + chunk pickle to rag/index/."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_all_chunks()
    print(f"Loaded {len(chunks)} chunks from {KNOWLEDGE}")

    print(f"Embedding with {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    vectors = np.asarray(vectors, dtype=np.float32)

    # Normalize so inner-product search equals cosine similarity.
    faiss.normalize_L2(vectors)

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
    print('Next: python -m rag.query "Which channels are Target YES?"')


if __name__ == "__main__":
    main()
