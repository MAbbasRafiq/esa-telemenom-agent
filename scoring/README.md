# `scoring/` — Telemanom inference + ESA metrics

Frozen lab model lives in `../model/`. This package does not retrain.

| File | Role (read in this order) |
|------|---------------------------|
| `windows.py` | Build LSTM input windows; apply means/stds |
| `errors.py` | \|pred−actual\| → smooth → Telemanom threshold → alarm intervals |
| `engine.py` | Load `model.h5`, run predict, return scores/alarms |
| `eval_labels.py` | Simple interval overlap vs `is_anomaly_*` (true/wrong/miss) |
| `esa_metrics.py` | TimeEval ESAScores (EW / AFF / alarming / F0.5) |
| `__init__.py` | Public API used by the agent tools |

The agent calls this library via `score_telemetry` and `compare_alarms_to_labels`.
