# Experiments

## C-MAPSS Validation

| Subset | Best Model | RMSE | R² | vs LSTM | vs DCNN |
|--------|-----------|------|-----|---------|---------|
| FD001 | HistGB / ET | 45.5 | 0.40 | 3.6x | 4.4x |
| FD002 | ET | 49.4 | 0.40 | 2.2x | 3.0x |
| FD003 | RF | 68.6 | 0.34 | 4.0x | 5.9x |
| FD004 | ET | 70.7 | 0.41 | 2.5x | 3.7x |

Single-row tree models (no sliding windows). Gap to LSTM/DCNN is expected — C-MAPSS is a time-series benchmark and temporal architectures carry the advantage.

## Ablation Study

See [`ablations/README.md`](../ablations/README.md). Key finding: Stacking beats individual models by ~18%.
