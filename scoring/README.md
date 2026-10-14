# `scoring/` — thin Telemanom infer (inside this repo)

No HTML. No TimeEval. No parent `telemanom_infer` import.

| File | Role (read in this order) |
|------|---------------------------|
| `windows.py` | Build LSTM input windows; apply means/stds |
| `errors.py` | \|pred−actual\| → smooth → Telemanom threshold → alarm intervals |
| `engine.py` | Load `../model/model.h5`, run predict, return scores/alarms |
| `__init__.py` | Public API: `TelemanomInferencer`, `write_outputs` |

CLI entrypoint: `../step3_score/score.py`
