"""Validation-set threshold calibration for alert decisions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.episode_replay import Trace
from src.eval.metrics import traces_to_frame

ROOT = Path(__file__).resolve().parents[2]
THRESH_PATH = ROOT / "data" / "processed" / "calibrated_thresholds.json"


def _f_beta(precision: float, recall: float, beta: float = 1.0) -> float:
    if precision + recall <= 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall + 1e-12)


def score_threshold(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    alert = (p >= thr).astype(int)
    tp = int(((alert == 1) & (y == 1)).sum())
    fp = int(((alert == 1) & (y == 0)).sum())
    fn = int(((alert == 0) & (y == 1)).sum())
    tn = int(((alert == 0) & (y == 0)).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    fpr = fp / max(fp + tn, 1)
    return {
        "threshold": float(thr),
        "recall": recall,
        "precision": precision,
        "fpr": fpr,
        "f1": _f_beta(precision, recall, 1.0),
        "f2": _f_beta(precision, recall, 2.0),
    }


def calibrate_threshold_from_probs(y: np.ndarray, p: np.ndarray, grid: np.ndarray | None = None) -> dict:
    """Pick operating threshold with false-alarm control.

    Priority:
    1) best F1 among thresholds with FPR <= 0.20
    2) else best F1 among precision >= 0.45
    3) else best F1 overall
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    if grid is None:
        qs = np.unique(np.quantile(p, np.linspace(0.05, 0.98, 30)))
        grid = np.unique(np.concatenate([qs, np.linspace(float(np.min(p)), float(np.max(p)), 25)]))

    scored = [score_threshold(y, p, float(thr)) for thr in grid]

    def pick(cands: list[dict]) -> dict | None:
        if not cands:
            return None
        return max(cands, key=lambda s: (s["f1"], s["recall"], -s["fpr"]))

    # Require useful discrimination: FPR controlled and F1 > 0
    best = pick([s for s in scored if s["fpr"] <= 0.20 and s["f1"] > 0])
    if best is None:
        best = pick([s for s in scored if s["precision"] >= 0.45 and s["f1"] > 0])
    if best is None:
        best = pick([s for s in scored if s["f1"] > 0])
    if best is None:
        best = pick(scored)
    assert best is not None
    return best


def calibrate_from_traces(traces: list[Trace], budget: int = 2) -> dict:
    df = traces_to_frame(traces)
    df = df[df["budget"] == budget]
    out: dict[str, dict] = {}
    for policy, g in df.groupby("policy"):
        y = g["severe_delay"].to_numpy()
        p = g["p_severe"].to_numpy()
        out[str(policy)] = calibrate_threshold_from_probs(y, p)
    THRESH_PATH.write_text(json.dumps({"budget": budget, "policies": out}, indent=2), encoding="utf-8")
    return out


def load_thresholds() -> dict | None:
    if not THRESH_PATH.exists():
        return None
    return json.loads(THRESH_PATH.read_text(encoding="utf-8"))
