"""Run four-policy ablation on held-out NYC episodes."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bn.disagreement import priors_from_episodes
from src.bn.inference import DelayRiskBN
from src.data.reason_mapper import factor_ids
from src.eval.episode_replay import replay_episode
from src.eval.scoreboard import write_scoreboard
from src.policy.consensus_voi import ConsensusVoIPolicy
from src.policy.disagreement_voi import DisagreementVoIPolicy
from src.policy.fixed_ranking import FixedRankingPolicy
from src.policy.static_bn import StaticBNPolicy


def main(max_pids: int | None = None, budgets=(1, 2, 3, 4)) -> None:
    ep_path = ROOT / "data" / "processed" / "episodes.csv"
    if not ep_path.exists():
        raise SystemExit("episodes.csv missing — run: python -m src.data.download_datasets && python -m src.data.nyc_ingest")

    episodes = pd.read_csv(ep_path)
    factors = factor_ids()
    priors = priors_from_episodes(episodes, factors)

    # Fixed order: descending consensus prior
    order = sorted(factors, key=lambda f: priors.consensus[f], reverse=True)

    policies_cons = [
        FixedRankingPolicy(order),
        StaticBNPolicy(order),
        ConsensusVoIPolicy(lookahead=1),
    ]
    policy_dis = DisagreementVoIPolicy(lookahead=1)

    test_pids = episodes.loc[episodes["split"] == "test", "pid"].unique()
    if max_pids is not None:
        test_pids = test_pids[:max_pids]
    print(f"[ablation] evaluating {len(test_pids)} test PIDs × budgets {list(budgets)} × 4 policies")
    traces = []
    for i, pid in enumerate(test_pids, start=1):
        if i == 1 or i % 50 == 0 or i == len(test_pids):
            print(f"[ablation] progress {i}/{len(test_pids)}")
        ep = episodes[episodes["pid"] == pid]
        for B in budgets:
            bn_c = DelayRiskBN(factor_priors=priors.consensus)
            for pol in policies_cons:
                traces.append(replay_episode(pol, bn_c, ep, B, factors))
            bn_d = DelayRiskBN(factor_priors=priors.disagreement)
            traces.append(replay_episode(policy_dis, bn_d, ep, B, factors))

    board = write_scoreboard(traces, out_name="scoreboard_full.csv")
    # also keep canonical scoreboard.csv as the full run
    board.to_csv(ROOT / "results" / "scoreboard.csv", index=False)
    print(board.to_string(index=False))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--max-pids", type=int, default=None, help="Limit test PIDs (default: all)")
    args = p.parse_args()
    main(max_pids=args.max_pids)
