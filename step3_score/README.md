# Step 3 — Thin Telemanom scoring CLI

```text
CSV → ../model/ + ../scoring/ → scores.csv + alarms.csv
```

No HTML. Scoring logic lives in **`../scoring/`** (not parent `telemanom_infer`).

## Run (after `conda activate esa-agent`)

```bash
cd esa-telemanom-agent/step3_score

python score.py \
  --input ../../data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.test.csv \
  --output-dir ./out \
  --skip-rows 31000 \
  --max-rows 3000
```

See root `README.md` for conda `esa-agent` setup.
