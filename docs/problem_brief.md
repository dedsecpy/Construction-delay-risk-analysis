# Problem brief

**Title:** Budget-Constrained Sequential Inspection for Construction Delay Risk Assessment Using a Bayesian Decision Network With Expert Disagreement

## Questions
- **Q1:** Does decision-oriented VoI beat fixed ranking at equal inspection budget?
- **Q2:** Does preserving disagreement beat a consensus prior?
- **Q3:** Does disagreement-aware VoI beat a static BN on budgeted detection metrics?

## Hypotheses
- **H1:** ConsensusVoI > Fixed (Recall@B, inspections-to-detection)
- **H2:** DisagreementVoI ≥ ConsensusVoI (especially high-disagreement regimes)
- **H3:** DisagreementVoI > StaticBN (Recall@B, lead-time, ROC-AUC after B=2)

## Outcome
Severe schedule delay: relative cumulative slip ≥ 25% of planned duration when available; otherwise top-quartile cumulative `variance_day` / schedule-change proxy (see dataset card).

## Strategies (matched budget)
1. Fixed ranking
2. Static BN (one-shot)
3. Consensus BN + VoI
4. Disagreement-aware BN + VoI (proposed)

## Novelty boundary
Integration of budgeted sequential inspection with disagreement-aware BN for construction severe-delay early warning on real multi-period capital-project data — not inventing BN or VoI alone.