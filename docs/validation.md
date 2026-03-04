# Validation

## Primary Split Policy

Rolling by season with holdout seasons configured in `configs/train.yaml`.

For each holdout season `S`:
- Train on all rows with `Season < S`
- Validate on all rows with `Season == S`

## Leakage Controls

- Rolling features are computed on shifted game history.
- Team snapshots are produced from regular season games only.
- Tournament outcomes are used only as targets, never as feature inputs.

## Reported Metrics

- Brier per base model per holdout season.
- Equal-blend baseline for diagnostics.
- OOF Brier for stacked model (raw and calibrated).
