# Metric scoreboard

All methods share the same held-out NYC PIDs, budgets `B∈{1,2,3,4}`, and observation model.

| Metric | Target |
|---|---|
| Recall@2 vs Fixed | +15–25% relative |
| Recall@2 vs StaticBN | +10–20% relative |
| Inspections-to-first-detection | ≥20% fewer vs Fixed; ≥15% vs StaticBN |
| ROC-AUC after B=2 | aim +0.03–0.08 vs StaticBN |
| Precision / FPR | Pareto improve or hold precision within 5 pp at matched recall |
| Brier / log-loss | better under high-disagreement stratum |

Always report absolute scores + relative % + 95% bootstrap CI.

## Claim hygiene
Do not claim published accuracy gains from unrelated datasets (e.g. Khodabakhshian +18%) unless the task and data match.