# DRIN evaluation figures

Generated from held-out **TEST** PIDs (`n=550`) unless noted. Re-run:

```bash
python experiments/generate_eval_figures.py
```

| File | Content |
|------|---------|
| `00_scorecard_overview.png` | Executive metric scorecard |
| `01_literature_accuracy_beat.png` | Accuracy vs Khodabakhshian / Tagharobi |
| `02_classifier_metric_bars.png` | Accuracy, balanced acc, P/R/Spec/FPR, F1/F2, AUC, PR-AUC, Brier, log-loss |
| `03_roc_pr_curves.png` | ROC + Precision–Recall |
| `04_confusion_matrix.png` | Confusion matrix with accuracy |
| `05_calibration_curve.png` | Calibration + Brier / log-loss / ECE |
| `06_decision_curve_net_benefit.png` | Decision-curve net benefit |
| `07_bootstrap_ci_and_pid_split.png` | 95% bootstrap CIs + 70/15/15 PID split |
| `08_literature_comparison_table.png` | Literature comparison table |
| `09_recall_precision_auc_at_B.png` | Recall@B / Precision@B / AUC@B |
| `10_inspections_to_first_detection.png` | Inspections-to-first-detection |
| `11_lead_time_to_alert.png` | Lead time to alert (early-warning) |
| `12_reason_factor_mapping_f1.png` | NLP reason→factor F1 (VAL) |
| `13_expected_utility_loss.png` | Expected utility / realized loss @ B=2 |

Summary JSON: `../eval_figures_summary.json`
