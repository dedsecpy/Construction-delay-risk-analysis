"""Assemble full scoreboard CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eval.bootstrap import bootstrap_recall
from src.eval.metrics import summarize, traces_to_frame
from src.eval.episode_replay import Trace

ROOT = Path(__file__).resolve().parents[2]


def write_scoreboard(traces: list[Trace], out_name: str = "scoreboard.csv") -> pd.DataFrame:
    df = traces_to_frame(traces)
    summary = summarize(df)
    # attach recall CIs for budget=2
    cis = []
    for policy in summary["policy"].unique():
        mean, lo, hi = bootstrap_recall(df, policy, budget=2)
        cis.append({"policy": policy, "budget": 2, "recall_mean_boot": mean, "recall_ci_lo": lo, "recall_ci_hi": hi})
    ci_df = pd.DataFrame(cis)
    out = summary.merge(ci_df, on=["policy", "budget"], how="left")
    dest = ROOT / "results" / out_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    df.to_csv(ROOT / "results" / "traces.csv", index=False)
    return out
