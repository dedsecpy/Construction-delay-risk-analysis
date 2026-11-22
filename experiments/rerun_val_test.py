"""Re-calibrate thresholds and re-run TEST using existing learned artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bn.learn_cpts import load_learned_cpts
from experiments.run_improved_pipeline import stage_calibrate, stage_test

RESULTS = ROOT / "results"


def main() -> None:
    episodes = pd.read_csv(ROOT / "data" / "processed" / "episodes.csv")
    learned = load_learned_cpts()
    if learned is None:
        raise SystemExit("learned_cpts.json missing — run improved pipeline first")
    thr_map = stage_calibrate(episodes, learned)
    board = stage_test(episodes, learned, thr_map)
    summary = {
        "learned": {"p_cause": learned["p_cause"], "leak": learned["leak"], "train_auc": learned.get("train_auc")},
        "thresholds": thr_map,
        "scoreboard": str(RESULTS / "scoreboard_improved.csv"),
    }
    (RESULTS / "improved_pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
