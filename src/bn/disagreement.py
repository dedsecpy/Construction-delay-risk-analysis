"""Consensus vs disagreement prior builders from empirical factor frequencies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PriorBundle:
    consensus: dict[str, float]
    disagreement: dict[str, float]
    vote_entropy: dict[str, float]


def _entropy_binary(p: float) -> float:
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def priors_from_episodes(episodes: pd.DataFrame, factors: list[str]) -> PriorBundle:
    """Estimate P(factor present) from train episodes; disagreement widens uncertain factors."""
    train = episodes[episodes["split"] == "train"] if "split" in episodes.columns else episodes
    if "observed" in train.columns:
        train = train[train["observed"] == 1]
    consensus: dict[str, float] = {}
    disagreement: dict[str, float] = {}
    ent: dict[str, float] = {}

    pid_factor = (
        train.groupby(["pid", "factor_id"]).size().reset_index(name="n")
        if len(train)
        else pd.DataFrame(columns=["pid", "factor_id", "n"])
    )

    for f in factors:
        # fraction of PIDs where factor appears at least once
        if len(pid_factor):
            pids = train["pid"].nunique()
            present_pids = pid_factor.loc[pid_factor["factor_id"] == f, "pid"].nunique()
            p = present_pids / max(pids, 1)
        else:
            p = 0.2
        p = float(np.clip(p, 0.05, 0.95))
        consensus[f] = p
        h = _entropy_binary(p)
        ent[f] = h
        # Disagreement prior: pull toward 0.5 proportional to entropy (preserve uncertainty)
        disagreement[f] = float(np.clip(0.5 * h + p * (1 - h), 0.05, 0.95))

    return PriorBundle(consensus=consensus, disagreement=disagreement, vote_entropy=ent)


def collapse_votes(vote_counts: dict[str, dict[str, int]]) -> dict[str, float]:
    """Majority/mean collapse: P(present) from Low/Med/High style counts if provided."""
    out: dict[str, float] = {}
    for factor, counts in vote_counts.items():
        low = counts.get("low", 0)
        med = counts.get("medium", counts.get("med", 0))
        high = counts.get("high", 0)
        total = low + med + high
        if total == 0:
            out[factor] = 0.2
            continue
        # map L/M/H to 0, 0.5, 1 presence weights
        out[factor] = (0.0 * low + 0.5 * med + 1.0 * high) / total
    return out


def disagreement_from_votes(vote_counts: dict[str, dict[str, int]]) -> dict[str, float]:
    collapsed = collapse_votes(vote_counts)
    out: dict[str, float] = {}
    for factor, counts in vote_counts.items():
        total = sum(counts.values()) or 1
        probs = np.array([counts.get("low", 0), counts.get("medium", counts.get("med", 0)), counts.get("high", 0)], dtype=float)
        probs = probs / total
        # Shannon entropy over 3 bins, normalized
        nz = probs[probs > 0]
        h = float(-(nz * np.log2(nz)).sum() / np.log2(3))
        p = collapsed[factor]
        out[factor] = float(np.clip(0.5 * h + p * (1 - h), 0.05, 0.95))
    return out
