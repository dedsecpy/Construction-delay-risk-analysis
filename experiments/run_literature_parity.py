"""Relative gains table vs Fixed and StaticBN + UK GMPP summary."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.uk_gmpp_ingest import summarize as uk_summarize


def main() -> None:
    board_path = ROOT / "results" / "scoreboard.csv"
    if not board_path.exists():
        raise SystemExit("Run experiments/run_ablation.py first")
    board = pd.read_csv(board_path)
    b2 = board[board["budget"] == 2].set_index("policy")
    base_fixed = b2.loc["fixed", "recall"] if "fixed" in b2.index else float("nan")
    base_static = b2.loc["static_bn", "recall"] if "static_bn" in b2.index else float("nan")
    rows = []
    for policy, row in b2.iterrows():
        rows.append(
            {
                "policy": policy,
                "recall@2": row["recall"],
                "rel_vs_fixed_%": (
                    100 * (row["recall"] - base_fixed) / base_fixed
                    if base_fixed and base_fixed > 0
                    else (None if row["recall"] == base_fixed else float("inf"))
                ),
                "rel_vs_static_%": (
                    100 * (row["recall"] - base_static) / base_static
                    if base_static and base_static > 0
                    else (None if row["recall"] == base_static else float("inf"))
                ),
                "recall_delta_vs_fixed": row["recall"] - base_fixed,
                "recall_delta_vs_static": row["recall"] - base_static,
                "roc_auc": row.get("roc_auc"),
                "mean_steps_to_alert": row.get("mean_steps_to_alert"),
            }
        )
    gains = pd.DataFrame(rows)
    out = ROOT / "results" / "gains_vs_baselines.csv"
    gains.to_csv(out, index=False)
    print(gains.to_string(index=False))
    try:
        uk = uk_summarize()
        print("UK GMPP:", uk)
    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    main()
