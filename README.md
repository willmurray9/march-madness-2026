# March Madness 2026 Kaggle Pipeline

End-to-end machine learning pipeline for the March Machine Learning Mania 2026 competition.

## Project Goals

- Minimize Brier score for all possible 2026 tournament matchups.
- Train separate men/women models with a shared framework.
- Keep runs reproducible, leakage-safe, and fast to iterate.

## Repository Layout

- `COMPETITION.md`: competition reference and data dictionary.
- `configs/`: pipeline configuration for data, features, models, and train/validation.
- `src/mm2026/`: package source code.
- `docs/`: methodology, validation policy, and results notes.
- `tests/`: unit and lightweight integration tests.
- `artifacts/`: generated model/report/submission outputs (gitignored).

## Quickstart

1. Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

2. Put Kaggle files in `data/raw/latest/`.

3. Run pipeline:

```bash
make data
make features
make train
make validate
make submit
make explain
make observe
```

4. Find outputs in:

- `artifacts/models/`
- `artifacts/reports/`
- `artifacts/submissions/`

## Observability Dashboard

Build/update observability artifacts:

```bash
make observe
```

Launch the dashboard:

```bash
make dashboard
```

The dashboard reads:

- `artifacts/reports/observability_latest.json`
- `artifacts/reports/runs_index.csv`
- per-gender train reports and latest submission manifest

## Modeling Approach

- Base models:
  - Logistic regression
  - Histogram gradient boosting
  - XGBoost
  - CatBoost
  - Elo probability model
- Stacking:
  - Logistic meta-model trained on out-of-fold base predictions
- Calibration:
  - Auto-select from `none`, `platt`, `isotonic` by OOF Brier
- Explainability:
  - Holdout-only permutation importance
  - Logistic coefficient rankings
  - SHAP plots for tree models in the Streamlit dashboard

## Validation Strategy

- Rolling season holdouts (default: 2021-2025)
- Train on seasons `< holdout`, validate on `holdout`
- Promotion uses mean/worst-season Brier gates

## Submission Rules Enforced

- Single file containing men + women predictions
- Exact `ID,Pred` schema
- Full ID coverage against `SampleSubmissionStage2.csv`
- Probabilities clipped to `[0, 1]`
