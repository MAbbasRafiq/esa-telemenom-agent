# Prepared telemetry samples

These CSV files are **small copies** cut from the original Mission1
`84_months.test.csv`. The original file is **not** modified.

| File | Friendly name | About |
|------|---------------|--------|
| `demo_known_events.csv` | Window with known events | ~3000 rows; good for compare-to-labels demos |
| `mid_mission_sample.csv` | Mid-mission sample | ~2500 rows |
| `late_mission_sample.csv` | Later mission sample | ~3500 rows |

Analysts can upload any of these (or their own CSV) in the UI.
The agent scores the **whole uploaded file** (no skip_rows needed).
