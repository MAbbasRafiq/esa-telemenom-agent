"""Friendly catalog of prepared sample CSV files under samples/.

These are small copies of slices from the original Mission1 test file.
The original full CSV is never changed.
"""
from __future__ import annotations

from pathlib import Path

# Project root = parent of api/
ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "samples"

# id → metadata shown to the UI (no row numbers / skip values)
SAMPLE_CATALOG = [
    {
        "id": "demo_known_events",
        "title": "Window with known events",
        "description": "A short prepared window that is useful for checking detections against known events.",
        "filename": "demo_known_events.csv",
    },
    {
        "id": "mid_mission_sample",
        "title": "Mid-mission sample",
        "description": "Another short prepared window from the middle of the mission.",
        "filename": "mid_mission_sample.csv",
    },
    {
        "id": "late_mission_sample",
        "title": "Later mission sample",
        "description": "A slightly longer prepared window from later in the mission.",
        "filename": "late_mission_sample.csv",
    },
]


def list_available_samples() -> list[dict]:
    """Return catalog entries that actually exist on disk."""
    available = []
    for item in SAMPLE_CATALOG:
        path = SAMPLES_DIR / item["filename"]
        if path.is_file():
            # Copy so we do not mutate the module-level list.
            entry = dict(item)
            entry["exists"] = True
            available.append(entry)
    return available


def resolve_sample_path(sample_id: str) -> Path:
    """Map a sample id to an absolute CSV path, or raise ValueError."""
    for item in SAMPLE_CATALOG:
        if item["id"] == sample_id:
            path = SAMPLES_DIR / item["filename"]
            if not path.is_file():
                raise ValueError(f"Sample file missing on server: {path.name}")
            return path
    raise ValueError(f"Unknown sample id: {sample_id}")
