"""ESA-ADB benchmark metrics (same code path as TimeEval lab).

Uses ``ESAScores`` from the parent ESA_ADB ``timeeval`` tree:
  alarming_precision, EW_precision, EW_recall, EW_F_0.50,
  AFF_precision, AFF_recall, AFF_F_0.50

Prediction rule matches TimeEval experiments:
  Score[t] = max over target channels of alarm[t]  (any-channel alarm).
"""
from __future__ import annotations

import contextlib
import io
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

# Parent ESA_ADB repo (…/ESA_ADB/esa-telemanom-agent/scoring → …/ESA_ADB)
ESA_ADB_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_CSV = ESA_ADB_ROOT / "data" / "ESA-Mission1" / "labels.csv"
DEFAULT_ANOMALY_TYPES = ESA_ADB_ROOT / "data" / "ESA-Mission1" / "anomaly_types.csv"


def _import_esa_scores():
    """Import ESAScores without loading full timeeval (avoids docker dep)."""
    if "timeeval.metrics.ESA_ADB_metrics" in sys.modules:
        from timeeval.metrics.ESA_ADB_metrics import ESAScores

        return ESAScores

    te_root = ESA_ADB_ROOT / "timeeval"
    if not te_root.is_dir():
        raise ImportError(f"timeeval not found at {te_root}")

    for name, path in [
        ("timeeval", te_root),
        ("timeeval.metrics", te_root / "metrics"),
    ]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]
            sys.modules[name] = mod

    from timeeval.metrics.ESA_ADB_metrics import ESAScores

    return ESAScores


def _load_labels_with_types(
    labels_csv: Path,
    anomaly_types_csv: Path,
    channels: Sequence[str],
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
) -> pd.DataFrame:
    """Load labels.csv + anomaly_types categories; keep events overlapping [t_min, t_max]."""
    labels = pd.read_csv(labels_csv, parse_dates=["StartTime", "EndTime"])
    labels["StartTime"] = labels["StartTime"].apply(
        lambda t: t.tz_localize(None) if getattr(t, "tzinfo", None) else t
    )
    labels["EndTime"] = labels["EndTime"].apply(
        lambda t: t.tz_localize(None) if getattr(t, "tzinfo", None) else t
    )

    types_df = pd.read_csv(anomaly_types_csv)
    # Last four columns are Category, Dimensionality, Locality, Length (ESA layout).
    type_cols = list(types_df.columns[-4:])
    for col in type_cols:
        labels[col] = ""
    for _, row in types_df.iterrows():
        labels.loc[labels["ID"] == row["ID"], type_cols] = row[type_cols].values

    labels = labels[labels["Channel"].isin(list(channels))]
    # Keep events that overlap the scored window, then clip to the window
    # so ESAScores full_range asserts hold (labels cannot extend past preds).
    labels = labels[(labels["EndTime"] >= t_min) & (labels["StartTime"] <= t_max)].copy()
    labels["StartTime"] = labels["StartTime"].clip(lower=t_min)
    labels["EndTime"] = labels["EndTime"].clip(upper=t_max)
    labels = labels[labels["StartTime"] < labels["EndTime"]]
    return labels.reset_index(drop=True)


def _alarms_to_global_pred(alarms: pd.DataFrame) -> np.ndarray:
    """Any-channel alarm at time t → Score=1 (TimeEval global rule)."""
    ts = pd.to_datetime(alarms.index)
    score = (alarms.to_numpy() > 0.5).any(axis=1).astype(np.uint8)
    # Object array of [Timestamp, score] pairs (ESAScores / convert_time_series_to_events).
    return np.array([[ts[i], int(score[i])] for i in range(len(ts))], dtype=object)


def compute_esa_scores(
    alarms: pd.DataFrame,
    labels_csv: Optional[PathLike] = None,
    anomaly_types_csv: Optional[PathLike] = None,
    beta: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Run ESAScores for (Rare Event+Anomaly) and (Anomaly-only), like the lab scripts.

    Returns
    -------
    dict with keys ``rare_event_and_anomaly`` and ``anomaly``, each a metric dict.
    """
    ESAScores = _import_esa_scores()

    labels_path = Path(labels_csv) if labels_csv else DEFAULT_LABELS_CSV
    types_path = Path(anomaly_types_csv) if anomaly_types_csv else DEFAULT_ANOMALY_TYPES
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")
    if not types_path.is_file():
        raise FileNotFoundError(f"anomaly_types.csv not found: {types_path}")

    # Keep pandas.Timestamp (not datetime.datetime) so portion intervals
    # use Timedelta with .value — required by ESAScores TNR correction.
    t_min = pd.Timestamp(alarms.index.min())
    t_max = pd.Timestamp(alarms.index.max())
    full_range = (t_min, t_max)

    y_true = _load_labels_with_types(
        labels_path, types_path, list(alarms.columns), t_min, t_max
    )
    y_pred = _alarms_to_global_pred(alarms)

    filters = [
        ("rare_event_and_anomaly", {"Category": ["Rare Event", "Anomaly"]}),
        ("anomaly", {"Category": ["Anomaly"]}),
    ]

    out: Dict[str, Dict[str, float]] = {}
    for name, select in filters:
        metric = ESAScores(betas=beta, select_labels=select, full_range=full_range)
        # ESAScores prints debug lists; silence that for the agent UI.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # Drop Channel column for global event scoring (TimeEval does this).
            scores = metric.score(y_true.drop(columns=["Channel"]), y_pred)
        out[name] = {k: float(v) if v == v else float("nan") for k, v in scores.items()}

    out["_meta"] = {
        "n_label_rows": float(len(y_true)),
        "n_unique_event_ids": float(y_true["ID"].nunique()) if len(y_true) else 0.0,
        "n_pred_alarm_timesteps": float(int(y_pred[:, 1].sum())),
        "t_min": t_min.isoformat(),
        "t_max": t_max.isoformat(),
    }
    return out


def format_esa_scores_report(scores: Dict[str, Dict[str, float]]) -> str:
    """Human/LLM-readable summary of ESAScores (lab-comparable names)."""
    meta = scores.get("_meta", {})
    lines: List[str] = [
        "ESA-ADB benchmark metrics (TimeEval ESAScores — same definitions as the lab).",
        "Global prediction: Score[t]=1 if ANY target channel alarms at t.",
        f"Window: {meta.get('t_min')} → {meta.get('t_max')}",
        f"Label rows in window: {int(meta.get('n_label_rows', 0))}  "
        f"unique event IDs: {int(meta.get('n_unique_event_ids', 0))}  "
        f"alarm timesteps: {int(meta.get('n_pred_alarm_timesteps', 0))}",
        "",
    ]

    sections = [
        ("rare_event_and_anomaly", "Category filter: Rare Event + Anomaly"),
        ("anomaly", "Category filter: Anomaly only"),
    ]
    keys = [
        "alarming_precision",
        "EW_precision",
        "EW_recall",
        "EW_F_0.50",
        "AFF_precision",
        "AFF_recall",
        "AFF_F_0.50",
    ]

    for key, title in sections:
        block = scores.get(key) or {}
        lines.append(title)
        for k in keys:
            if k in block:
                lines.append(f"  {k}={block[k]:.6f}")
        lines.append("")

    lines.append(
        "Note: EW_precision includes the Sehili time-wise false-positive correction; "
        "these numbers are comparable in *definition* to the lab Telemanom report "
        "(a short demo slice is still not the full 84_months run)."
    )
    return "\n".join(lines).rstrip()
