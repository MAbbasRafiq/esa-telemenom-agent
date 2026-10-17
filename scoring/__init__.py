"""Telemanom-ESA scoring + ESA-ADB metrics.

Modules to read for understanding (in order):
  1. windows.py  — sliding windows + normalize
  2. errors.py   — prediction error + Telemanom dynamic threshold
  3. engine.py   — load model.h5, run predict, write scores/alarms
  4. eval_labels.py — simple interval overlap (true/wrong/miss)
  5. esa_metrics.py — TimeEval ESAScores (EW / AFF / alarming / F0.5)

Artifacts live in ../model/ (passed as artifacts_dir).
"""

from .engine import InferenceResult, TelemanomInferencer, write_outputs  # noqa: F401
from .eval_labels import (  # noqa: F401
    compare_alarms_to_labels,
    load_labels_aligned,
)
from .esa_metrics import (  # noqa: F401
    compute_esa_scores,
    format_esa_scores_report,
)

__all__ = [
    "TelemanomInferencer",
    "InferenceResult",
    "write_outputs",
    "compare_alarms_to_labels",
    "load_labels_aligned",
    "compute_esa_scores",
    "format_esa_scores_report",
]
