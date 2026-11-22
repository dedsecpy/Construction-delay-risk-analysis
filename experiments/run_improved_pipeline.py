"""Full improved pipeline: NLP + learned CPTs + val-tuned thresholds → test results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bn.inference import DelayRiskBN
from src.bn.learn_cpts import learn_noisy_or_params, load_learned_cpts
from src.data.nlp_reason_classifier import evaluate_on_texts, train_nlp_classifier
from src.data.nyc_ingest import _normalize_history, build_episodes
from src.data.reason_mapper import factor_ids, map_reason_regex, reset_nlp_cache
from src.eval.calibrate import calibrate_from_traces, load_thresholds
from src.eval.episode_replay import replay_episode
from src.eval.scoreboard import write_scoreboard
from src.policy.consensus_voi import ConsensusVoIPolicy
from src.policy.disagreement_voi import DisagreementVoIPolicy
from src.policy.fixed_ranking import FixedRankingPolicy
from src.policy.static_bn import StaticBNPolicy

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def _assign_splits(pids: np.ndarray, seed: int = 42) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    pids = np.array(pids, dtype=str)
    rng.shuffle(pids)
    n = len(pids)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    out = {}
    for i, pid in enumerate(pids):
        if i < n_train:
            out[pid] = "train"
        elif i < n_train + n_val:
            out[pid] = "val"
        else:
            out[pid] = "test"
    return out


def stage_train_nlp(seed: int = 42) -> None:
    print("\n=== Stage 1: Train NLP reason classifier (train PIDs only) ===")
    hist = _normalize_history(pd.read_csv(RAW / "nyc_schedule_history.csv", low_memory=False))
    split_map = _assign_splits(hist["pid"].unique(), seed=seed)
    hist["pid"] = hist["pid"].astype(str)
    hist["split"] = hist["pid"].map(split_map)
    train_reasons = hist.loc[hist["split"] == "train", "reason"]
    model = train_nlp_classifier(train_reasons, random_state=seed)
    reset_nlp_cache()
    report = evaluate_on_texts(hist.loc[hist["split"] == "val", "reason"], model=model)
    (PROC / "nlp_val_report.txt").write_text(report, encoding="utf-8")
    print(report)


def stage_rebuild_episodes() -> pd.DataFrame:
    print("\n=== Stage 2: Rebuild episodes with NLP mapping ===")
    reset_nlp_cache()
    return build_episodes(seed=42, use_nlp=True)


def stage_learn_cpts(episodes: pd.DataFrame) -> dict:
    print("\n=== Stage 3: Learn CPTs / Noisy-OR from TRAIN ===")
    train = episodes[episodes["split"] == "train"]
    learned = learn_noisy_or_params(train)
    print(
        json.dumps(
            {
                "p_cause": learned["p_cause"],
                "leak": learned["leak"],
                "train_loglik": learned["train_loglik"],
                "severe_rate_train": learned["severe_rate_train"],
            },
            indent=2,
        )
    )
    return learned


def _make_policies(order: list[str], lookahead: int = 2):
    return [
        FixedRankingPolicy(order),
        StaticBNPolicy(order),
        ConsensusVoIPolicy(lookahead=lookahead),
        DisagreementVoIPolicy(lookahead=lookahead),
    ]


def _run_split(
    episodes: pd.DataFrame,
    split: str,
    learned: dict,
    budgets=(1, 2, 3, 4),
    thresholds: dict[str, float] | None = None,
    lookahead: int = 1,
    default_threshold: float = 0.65,
):
    factors = factor_ids()
    order = sorted(factors, key=lambda f: learned["consensus_priors"][f], reverse=True)
    policies = _make_policies(order, lookahead=lookahead)
    pids = episodes.loc[episodes["split"] == split, "pid"].unique()
    print(f"[{split}] {len(pids)} PIDs, lookahead={lookahead}")
    traces = []
    for i, pid in enumerate(pids, start=1):
        if i == 1 or i % 50 == 0 or i == len(pids):
            print(f"[{split}] progress {i}/{len(pids)}", flush=True)
        ep = episodes[episodes["pid"] == pid]
        for B in budgets:
            for pol in policies:
                priors = (
                    learned["disagreement_priors"]
                    if pol.name == "disagreement_voi"
                    else learned["consensus_priors"]
                )
                bn = DelayRiskBN(
                    factor_priors=priors,
                    p_cause=learned["p_cause"],
                    leak=learned["leak"],
                )
                thr = default_threshold
                if thresholds and pol.name in thresholds:
                    thr = float(thresholds[pol.name])
                traces.append(
                    replay_episode(pol, bn, ep, B, factors, threshold=thr)
                )
    return traces


def stage_calibrate(episodes: pd.DataFrame, learned: dict) -> dict[str, float]:
    print("\n=== Stage 4: Validate + calibrate alert thresholds (B=2) ===")
    # Collect probabilities with a neutral mid threshold so p_severe is unrestricted
    val_traces = _run_split(
        episodes,
        split="val",
        learned=learned,
        budgets=(2,),
        thresholds=None,
        lookahead=1,
        default_threshold=0.0,  # never early-stop alert logic for calibration probs
    )
    # For calibration we need raw probabilities; recompute alerts later.
    # Force alerted unused: set threshold 1.1 equivalent by patching after
    calibrated = calibrate_from_traces(val_traces, budget=2)
    thr_map = {k: float(v["threshold"]) for k, v in calibrated.items()}
    print(json.dumps(calibrated, indent=2))
    return thr_map


def stage_test(episodes: pd.DataFrame, learned: dict, thr_map: dict[str, float]) -> pd.DataFrame:
    print("\n=== Stage 5: TEST evaluation with calibrated thresholds ===")
    traces = _run_split(
        episodes,
        split="test",
        learned=learned,
        budgets=(1, 2, 3, 4),
        thresholds=thr_map,
        lookahead=1,
        default_threshold=0.65,
    )
    board = write_scoreboard(traces, out_name="scoreboard_improved.csv")
    board.to_csv(RESULTS / "scoreboard.csv", index=False)

    # Gains at B=2
    b2 = board[board["budget"] == 2].set_index("policy")
    base_fixed = float(b2.loc["fixed", "recall"]) if "fixed" in b2.index else 0.0
    base_static = float(b2.loc["static_bn", "recall"]) if "static_bn" in b2.index else 0.0
    rows = []
    for policy, row in b2.iterrows():
        rows.append(
            {
                "policy": policy,
                "recall@2": row["recall"],
                "precision@2": row["precision"],
                "fpr@2": row["fpr"],
                "roc_auc@2": row["roc_auc"],
                "brier@2": row["brier"],
                "recall_delta_vs_fixed": row["recall"] - base_fixed,
                "recall_delta_vs_static": row["recall"] - base_static,
                "threshold": thr_map.get(policy),
            }
        )
    gains = pd.DataFrame(rows)
    gains.to_csv(RESULTS / "gains_improved.csv", index=False)
    print(board.to_string(index=False))
    print("\nGains @ B=2")
    print(gains.to_string(index=False))
    return board


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not (RAW / "nyc_schedule_history.csv").exists():
        from src.data.download_datasets import main as download_main

        download_main()

    stage_train_nlp(seed=42)
    episodes = stage_rebuild_episodes()
    learned = stage_learn_cpts(episodes)
    thr_map = stage_calibrate(episodes, learned)
    stage_test(episodes, learned, thr_map)

    summary = {
        "splits": json.loads((PROC / "splits.json").read_text(encoding="utf-8")),
        "learned_cpts": {
            "p_cause": learned["p_cause"],
            "leak": learned["leak"],
        },
        "thresholds": thr_map,
        "artifacts": {
            "episodes": str(PROC / "episodes.csv"),
            "learned_cpts": str(PROC / "learned_cpts.json"),
            "thresholds": str(PROC / "calibrated_thresholds.json"),
            "scoreboard": str(RESULTS / "scoreboard_improved.csv"),
            "gains": str(RESULTS / "gains_improved.csv"),
        },
    }
    (RESULTS / "improved_pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== DONE: improved train/val/test pipeline ===")


if __name__ == "__main__":
    main()
