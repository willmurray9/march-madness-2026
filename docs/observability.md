# Observability

This project includes a pipeline observability layer so each run can be inspected end-to-end.

## Artifacts

`make observe` writes:

- `artifacts/reports/observability_snapshot_<UTCSTAMP>.json`: immutable run snapshot
- `artifacts/reports/observability_latest.json`: latest snapshot alias
- `artifacts/reports/runs_index.csv`: append-only run history table

Each snapshot includes:

- Config file hashes for `configs/data.yaml`, `configs/features.yaml`, `configs/models.yaml`, `configs/train.yaml`
- Data and feature file summaries (exists, rows, cols, null fraction, modified time)
- Men and women train reports (`*_train_report.json`)
- Latest submission manifest (`*_manifest.json`)

## Dashboard

Run:

```bash
make dashboard
```

The Streamlit app renders:

- Run history and OOF metric trends
- Pipeline asset inventory and data quality summary
- Training metrics and season-level validation curves
- Blend/calibration details for each gender
- Latest submission prediction distribution
