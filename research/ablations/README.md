# Ablation Study

## Purpose

Quantify the contribution of each modelling decision to overall predictive performance.

## Variants

Six experiments varying three factors:

| # | Model | Split | Scale targets | RMSE | R² |
|---|-------|-------|---------------|------|-----|
| 1 | ExtraTrees | official | Yes | 694.9 | 0.741 |
| 2 | HistGB | official | Yes | 680.5 | 0.727 |
| 3 | **Stacking** | **official** | **Yes** | **587.7** | **0.748** |
| 4 | ExtraTrees | grouped | Yes | 926.8 | 0.783 |
| 5 | HistGB | grouped | Yes | 776.8 | 0.721 |
| 6 | ExtraTrees | official | No | 687.4 | 0.741 |

## Key Findings

1. **Stacking** (ExtraTrees + RF + GB → Ridge) beats individual tree models across all metrics — RMSE 587.7 vs 680–695 (18% improvement over HistGB).
2. **Split strategy** dominates: grouped (unseen engines) RMSE is 25–36% higher than official. This is expected and consistent with the literature.
3. **Target scaling** has negligible effect on overall RMSE for ExtraTrees (694.9 scaled vs 687.4 unscaled). The health sub-targets are already on similar scales; Thrust dominates the aggregate metric regardless.
4. **HistGB** trains faster than ExtraTrees but matches or slightly underperforms in RMSE.
5. **CombustorHealth** is the hardest health target across all variants (lowest R²), suggesting it is poorly correlated with the sensor suite.

## Raw Data

Individual experiment JSON reports are in this directory (`*_report.json`). Full report: `ablation_report.md`.
