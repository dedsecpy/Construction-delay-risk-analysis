"""Bootstrap confidence intervals for scoreboard rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import summarize


def bootstrap_recall(df: pd.DataFrame, policy: str, budget: int, n_boot: int = 200, seed: int = 0) -> tuple[float, float, float]:
    g = df[(df["policy"] == policy) & (df["budget"] == budget)]
    if g.empty:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    pids = g["pid"].unique()
    for _ in range(n_boot):
        sample_pids = rng.choice(pids, size=len(pids), replace=True)
        parts = [g[g["pid"] == pid] for pid in sample_pids]
        boot = pd.concat(parts, ignore_index=True)
        s = summarize(boot)
        vals.append(float(s["recall"].iloc[0]))
    arr = np.asarray(vals)
    return float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))
