"""Scoreboard metrics from replay traces."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from src.eval.episode_replay import Trace


def traces_to_frame(traces: list[Trace]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pid": t.pid,
                "policy": t.policy,
                "budget": t.budget,
                "severe_delay": t.severe_delay,
                "alerted": int(t.alerted),
                "p_severe": t.p_severe,
                "n_inspections": len(t.inspections),
                "steps_to_alert": t.steps_to_alert if t.steps_to_alert is not None else np.nan,
            }
            for t in traces
        ]
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (policy, budget), g in df.groupby(["policy", "budget"]):
        y = g["severe_delay"].to_numpy()
        alert = g["alerted"].to_numpy()
        p = g["p_severe"].to_numpy()
        tp = ((alert == 1) & (y == 1)).sum()
        fp = ((alert == 1) & (y == 0)).sum()
        fn = ((alert == 0) & (y == 1)).sum()
        tn = ((alert == 0) & (y == 0)).sum()
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        fpr = fp / max(fp + tn, 1)
        try:
            auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
        except ValueError:
            auc = np.nan
        try:
            brier = brier_score_loss(y, p)
        except ValueError:
            brier = np.nan
        rows.append(
            {
                "policy": policy,
                "budget": budget,
                "n": len(g),
                "recall": recall,
                "precision": precision,
                "fpr": fpr,
                "roc_auc": auc,
                "brier": brier,
                "mean_steps_to_alert": g["steps_to_alert"].mean(),
                "mean_p_severe": p.mean(),
            }
        )
    return pd.DataFrame(rows)
