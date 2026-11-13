"""Upload / CSV size guards with analyst-friendly error messages."""
from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, UploadFile

from config import MAX_TOOL_SCORE_ROWS, MAX_UI_SCORE_ROWS, MAX_UPLOAD_BYTES


def clamp_score_rows(max_rows: int | None, *, ui: bool = False) -> int | None:
    """Cap max_rows so a huge LLM/UI request cannot melt the CPU."""
    if max_rows is None:
        return None
    cap = MAX_UI_SCORE_ROWS if ui else MAX_TOOL_SCORE_ROWS
    try:
        n = int(max_rows)
    except (TypeError, ValueError):
        return cap
    if n < 1:
        return 1
    return min(n, cap)


def save_upload_checked(upload: UploadFile, uploads_dir: Path) -> Path:
    """Save a CSV upload after size checks. Raises HTTPException on bad input."""
    import uuid

    uploads_dir.mkdir(parents=True, exist_ok=True)

    original_name = upload.filename or "telemetry.csv"
    safe_name = Path(original_name).name
    if not safe_name.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV telemetry file.",
        )

    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(data) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        got = len(data) / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=(
                f"This CSV is too large ({got:.1f} MB). "
                f"Please upload a slice under {mb:.0f} MB "
                f"(about {MAX_UI_SCORE_ROWS:,} rows is a safe window)."
            ),
        )

    # Rough row count (header + data). Avoid full pandas parse on every upload.
    text = data.decode("utf-8", errors="replace")
    # Count newlines; last line without \\n still counts as a row if non-empty.
    lines = text.count("\n") + (0 if text.endswith("\n") or not text.strip() else 1)
    data_rows = max(0, lines - 1)
    if data_rows > MAX_UI_SCORE_ROWS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"This CSV has about {data_rows:,} rows. "
                f"Please upload a window of at most {MAX_UI_SCORE_ROWS:,} rows "
                "so anomaly checks stay practical on this machine."
            ),
        )

    dest = uploads_dir / f"{uuid.uuid4().hex}_{safe_name}"
    dest.write_bytes(data)
    return dest


def validate_existing_csv_rows(csv_path: str | None) -> None:
    """Optional light check when reusing a session CSV path (samples / prior upload)."""
    if not csv_path:
        return
    path = Path(csv_path)
    if not path.is_file():
        raise HTTPException(
            status_code=400,
            detail="The attached telemetry file is missing. Please upload it again.",
        )
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"Attached CSV exceeds the {mb:.0f} MB limit. Please use a smaller slice.",
        )
