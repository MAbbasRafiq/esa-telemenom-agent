"""Compare Telemanom alarms to ESA ground-truth labels (no HTML).

Labels live in the telemetry CSV as columns ``is_anomaly_<channel>``.
We use the same interval overlap rules as the older telemanom_infer report:
  true  = alarm segment overlaps a labelled segment
  wrong = alarm segment with no label overlap
  miss  = labelled segment with no alarm overlap
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def _intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def classify_channel(alarm: np.ndarray, label: np.ndarray) -> Dict[str, object]:
    alarm_b = np.asarray(alarm).astype(float) > 0.5
    label_b = np.asarray(label).astype(float) > 0
    alarm_iv = _intervals(alarm_b)
    label_iv = _intervals(label_b)

    n_true = n_wrong = 0
    for iv in alarm_iv:
        if any(_overlaps(iv, lv) for lv in label_iv):
            n_true += 1
        else:
            n_wrong += 1
    n_miss = sum(
        1 for lv in label_iv if not any(_overlaps(lv, av) for av in alarm_iv)
    )
    return {
        "n_true": n_true,
        "n_wrong": n_wrong,
        "n_miss": n_miss,
        "n_label": len(label_iv),
        "n_alarm": len(alarm_iv),
    }


def _skip_arg(skip_rows: int):
    return range(1, skip_rows + 1) if skip_rows > 0 else None


def load_labels_aligned(
    input_csv: PathLike,
    timestamps: pd.Index,
    channels: Sequence[str],
    max_rows: Optional[int] = None,
    skip_rows: int = 0,
) -> pd.DataFrame:
    path = Path(input_csv)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if "timestamp" not in header:
        raise ValueError(f"{path} has no timestamp column")
    label_cols = [f"is_anomaly_{ch}" for ch in channels if f"is_anomaly_{ch}" in header]
    usecols = ["timestamp"] + label_cols
    df = pd.read_csv(
        path,
        usecols=usecols,
        index_col="timestamp",
        parse_dates=True,
        skiprows=_skip_arg(skip_rows),
        nrows=max_rows,
    )
    df = df.reindex(timestamps)
    out = pd.DataFrame(0, index=timestamps, columns=list(channels), dtype=np.float32)
    for ch in channels:
        name = f"is_anomaly_{ch}"
        if name in df.columns:
            out[ch] = df[name].fillna(0).astype(np.float32)
    return out


def compare_alarms_to_labels(
    alarms: pd.DataFrame,
    labels: pd.DataFrame,
    top_n: int = 8,
) -> str:
    """Return a short text summary for an agent / human."""
    channels = list(alarms.columns)
    det_true = det_wrong = det_miss = det_label = det_alarm = 0
    other_alarm = 0
    n_labelled_ch = 0
    n_other_alarm_ch = 0
    labelled_rows = []

    for ch in channels:
        if ch not in labels.columns:
            continue
        cls = classify_channel(alarms[ch].values, labels[ch].values)
        has_label = int(cls["n_label"]) > 0
        has_alarm = int(cls["n_alarm"]) > 0
        if has_label:
            n_labelled_ch += 1
            det_true += int(cls["n_true"])
            det_wrong += int(cls["n_wrong"])
            det_miss += int(cls["n_miss"])
            det_label += int(cls["n_label"])
            det_alarm += int(cls["n_alarm"])
            labelled_rows.append(
                (
                    ch,
                    int(cls["n_true"]),
                    int(cls["n_wrong"]),
                    int(cls["n_miss"]),
                    int(cls["n_label"]),
                    int(cls["n_alarm"]),
                )
            )
        elif has_alarm:
            n_other_alarm_ch += 1
            other_alarm += int(cls["n_alarm"])

    # Precision is alarm-centric: among alarm events, how many hit a label?
    prec = det_true / det_alarm if det_alarm else float("nan")
    # Recall must be label-centric (always ≤ 1): among label events, how many
    # were hit by ≥1 alarm? Do NOT use true/labels — true counts *alarms*, so
    # several alarms can overlap one label and true/labels can exceed 1.
    labels_hit = det_label - det_miss
    rec = labels_hit / det_label if det_label else float("nan")

    lines = [
        "Alarm vs ground-truth labels (interval overlap on this slice).",
        "Definitions: true=alarm overlaps label; wrong=alarm with no label; miss=label with no alarm.",
        f"channels_total={len(channels)}  channels_with_any_label={n_labelled_ch}  "
        f"channels_alarm_but_no_label_here={n_other_alarm_ch}",
        f"On labelled channels: alarm_events={det_alarm}  label_events={det_label}  "
        f"true={det_true}  wrong={det_wrong}  miss={det_miss}",
        (
            f"event_precision(true/alarms)={prec:.3f}  "
            f"event_recall(labels_hit/labels)={rec:.3f}  "
            f"(labels_hit={labels_hit} = label_events - miss)"
        )
        if det_alarm or det_label
        else "No labelled events or alarms to score on this slice.",
        "Note: true can be > label_events when several alarm intervals overlap "
        "the same label; that is why recall uses labels_hit, not true/labels.",
        f"alarms_on_unlabelled_channels_events={other_alarm} "
        "(not counted as wrong — no label in this window to compare).",
    ]

    if labelled_rows:
        # Sort by misses + wrongs then by alarms
        labelled_rows.sort(key=lambda r: (r[3] + r[2], r[5]), reverse=True)
        lines.append(f"Top {min(top_n, len(labelled_rows))} labelled channels (true/wrong/miss | labels/alarms):")
        for ch, t, w, m, nl, na in labelled_rows[:top_n]:
            lines.append(f"  {ch}: true={t} wrong={w} miss={m} | labels={nl} alarms={na}")

    return "\n".join(lines)
