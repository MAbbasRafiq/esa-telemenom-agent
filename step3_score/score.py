#!/usr/bin/env python3
"""Step 3 — Thin Telemanom scoring CLI.

Uses:
  ../model/    — weights + means/stds + contract
  ../scoring/  — thin infer code (no HTML, no parent telemanom_infer)

  python score.py --input telemetry.csv --output-dir ./out --max-rows 3000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
sys.path.insert(0, str(ROOT))

from scoring import TelemanomInferencer, write_outputs  # noqa: E402


def summarize(result, top_n: int = 10, window_size: int = 250) -> str:
    """Short text summary for humans / a future agent tool."""
    lines = [
        f"rows={len(result.scores)}  target_channels={result.scores.shape[1]}",
        f"window_size={window_size} (first window rows have zero scores/alarms)",
    ]
    if result.alarms is None:
        lines.append("alarms: not computed (--no-threshold)")
        return "\n".join(lines)

    alarm_counts = result.alarms.sum(axis=0).sort_values(ascending=False)
    active = alarm_counts[alarm_counts > 0]
    lines.append(f"channels_with_any_alarm={len(active)} / {len(alarm_counts)}")
    if len(active) == 0:
        lines.append("No alarms in this slice.")
    else:
        lines.append(f"Top {min(top_n, len(active))} channels by alarm timesteps:")
        for ch, cnt in active.head(top_n).items():
            lines.append(f"  {ch}: {int(cnt)}")
    return "\n".join(lines)


def run(
    input_csv: Path,
    output_dir: Path,
    max_rows: int | None,
    skip_rows: int,
    threshold: bool,
) -> None:
    if not (MODEL_DIR / "model.h5").exists():
        raise SystemExit(f"Missing model pack under {MODEL_DIR}")

    print(f"model/:   {MODEL_DIR}")
    print(f"scoring/: {ROOT / 'scoring'}")
    print(f"input:    {input_csv}")
    clf = TelemanomInferencer(artifacts_dir=MODEL_DIR)
    result = clf.predict(
        input_csv,
        threshold=threshold,
        max_rows=max_rows,
        skip_rows=skip_rows,
    )
    write_outputs(result, output_dir)
    print("\n=== Summary ===")
    print(summarize(result, window_size=int(clf.contract["window_size"])))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: score telemetry with esa-telemanom-agent/model/ + scoring/"
    )
    parser.add_argument("--input", type=Path, required=True, help="Telemetry CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "out",
        help="Folder for scores.csv + alarms.csv (no HTML)",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-rows", type=int, default=0)
    parser.add_argument(
        "--no-threshold",
        action="store_true",
        help="Scores only (no Telemanom dynamic alarms)",
    )
    args = parser.parse_args()
    run(
        input_csv=args.input,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        skip_rows=args.skip_rows,
        threshold=not args.no_threshold,
    )


if __name__ == "__main__":
    main()
