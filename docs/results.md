# Results Log

Track each notable run with:

- Data snapshot tag/date
- Feature config hash or version note
- Model config note
- OOF Brier (raw + calibrated)
- Selected calibration method
- Submission artifact path

Recommended table columns:

| Run | Snapshot | Men Brier | Women Brier | Blend Notes | Promoted |
|---|---|---:|---:|---|---|
| `20260316T182337Z` | Kaggle refresh `2026-03-16` (regular season through `2026-03-15`) | 0.190165 | 0.133839 | `blend` + `isotonic`; `M=trend`, `W=baseline`; submission `artifacts/submissions/20260316T182328Z_stage2.csv` | Women: yes, Men: no |
