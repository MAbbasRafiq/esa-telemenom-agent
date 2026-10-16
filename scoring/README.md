# `scoring/` — thin Telemanom infer (inside this repo)

No HTML. No TimeEval. No parent `telemanom_infer` import.

| File | Role (read in this order) |
|------|---------------------------|
| `windows.py` | Build LSTM input windows; apply means/stds |
| `errors.py` | \|pred−actual\| → smooth → Telemanom threshold → alarm intervals |
| `engine.py` | Load `../model/model.h5`, run predict, return scores/alarms |
| `eval_labels.py` | Compare alarms vs `is_anomaly_*` labels (true/wrong/miss) |
| `__init__.py` | Public API |

CLI: `../step3_score/score.py` · Agent tool: `compare_alarms_to_labels` in `step4_agent/`
