"""Learn factor priors and Noisy-OR parameters from train episodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.bn.cpt import noisy_or_severe_table
from src.bn.disagreement import priors_from_episodes
from src.data.reason_mapper import factor_ids

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "processed" / "learned_cpts.json"


def _pid_factor_matrix(episodes: pd.DataFrame, factors: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pid_sev = episodes.groupby("pid")["severe_delay"].max()
    observed = episodes[episodes.get("observed", 1) == 1] if "observed" in episodes.columns else episodes
    present = (observed.groupby(["pid", "factor_id"]).size().unstack(fill_value=0) > 0).astype(int)
    for f in factors:
        if f not in present.columns:
            present[f] = 0
    present = present.reindex(pid_sev.index).fillna(0).astype(int)[factors]
    X = present.to_numpy(dtype=int)
    y = pid_sev.to_numpy(dtype=int)
    return X, y


def _predict_p(X: np.ndarray, p_cause: float, leak: float) -> np.ndarray:
    n_factors = X.shape[1]
    table = noisy_or_severe_table(n_factors, leak=leak, p_cause=p_cause)
    ps = np.zeros(len(X), dtype=float)
    for i in range(len(X)):
        cfg = 0
        for j, bit in enumerate(X[i]):
            if bit:
                cfg |= 1 << j
        ps[i] = table[cfg, 1]
    return ps


def learn_noisy_or_params(train_episodes: pd.DataFrame) -> dict:
    factors = factor_ids()
    # Only rows with observed evidence for presence stats
    X, y = _pid_factor_matrix(train_episodes, factors)
    best = {"p_cause": 0.4, "leak": 0.05, "score": -1.0, "auc": 0.5, "loglik": -1e18}

    for p_cause in np.linspace(0.25, 0.7, 10):
        for leak in np.linspace(0.02, 0.12, 6):
            p = _predict_p(X, float(p_cause), float(leak))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            ll = float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
            try:
                auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5
            except ValueError:
                auc = 0.5
            # Prefer AUC, then loglik
            score = auc * 1000 + ll / max(len(y), 1)
            if score > best["score"]:
                best = {
                    "p_cause": float(p_cause),
                    "leak": float(leak),
                    "score": float(score),
                    "auc": auc,
                    "loglik": ll,
                }

    # Presence priors from observed evidence only
    obs = train_episodes[train_episodes["observed"] == 1] if "observed" in train_episodes.columns else train_episodes
    if len(obs) == 0:
        obs = train_episodes
    # Temporarily mark split for priors helper
    obs = obs.copy()
    if "split" not in obs.columns:
        obs["split"] = "train"
    bundle = priors_from_episodes(obs, factors)

    # Cap other_admin prior so it cannot dominate
    for d in (bundle.consensus, bundle.disagreement):
        if d.get("other_admin", 0) > 0.35:
            d["other_admin"] = 0.35

    out = {
        "factors": factors,
        "p_cause": best["p_cause"],
        "leak": best["leak"],
        "train_auc": best["auc"],
        "train_loglik": best["loglik"],
        "consensus_priors": bundle.consensus,
        "disagreement_priors": bundle.disagreement,
        "vote_entropy": bundle.vote_entropy,
        "n_train_pids": int(train_episodes["pid"].nunique()),
        "severe_rate_train": float(train_episodes.groupby("pid")["severe_delay"].max().mean()),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def load_learned_cpts() -> dict | None:
    if not OUT_PATH.exists():
        return None
    return json.loads(OUT_PATH.read_text(encoding="utf-8"))
