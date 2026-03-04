# Methodology

## Data Flow

1. Ingest raw Kaggle CSV snapshots.
2. Normalize regular season games into long team-game format.
3. Build season-level team snapshots with leakage-safe rolling stats.
4. Build matchup-level delta features for tournament training rows and Stage 2 inference rows.

## Core Feature Families

- Team strength proxies: offense, defense, net efficiency, margin.
- Form windows: short/mid/long rolling means (shifted by one game).
- Elo: season-local rating from compact results.
- Matchup deltas: lower team minus higher team feature differences.

## Model Stack

- Base: logistic regression + histogram gradient boosting + Elo curve.
- Meta: logistic regression over base prediction outputs.
- Calibration: best Brier from none/platt/isotonic.
