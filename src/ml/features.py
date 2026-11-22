"""Project-level early-warning features (no label leakage)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.reason_mapper import factor_ids

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower().strip().replace(" ", "_").replace("(", "").replace(")", ""): c for c in df.columns}
    rename = {}
    for canon, cands in {
        "pid": ["pid"],
        "reporting_period": ["reporting_period"],
        "variance_day": ["variance_day"],
        "reason": ["reason_for_forecast_completion_change"],
        "phase": ["current_phase"],
        "managing_agency": ["managing_agency"],
    }.items():
        for cand in cands:
            if cand in lower:
                rename[lower[cand]] = canon
                break
    out = df.rename(columns=rename).copy()
    out["pid"] = out["pid"].astype(str)
    out["variance_day"] = pd.to_numeric(out.get("variance_day"), errors="coerce").fillna(0.0)
    out["reporting_period"] = pd.to_numeric(out.get("reporting_period"), errors="coerce")
    if "reason" not in out.columns:
        out["reason"] = ""
    if "phase" not in out.columns:
        out["phase"] = ""
    if "managing_agency" not in out.columns:
        out["managing_agency"] = ""
    return out.sort_values(["pid", "reporting_period"])


def build_early_warning_table(early_frac: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """Build features from the first early_frac of each PID's history; label from FULL history.

    This is a true early-warning task: predict eventual severe delay using only early signals.
    """
    from src.data.nyc_ingest import _severe_labels
    from src.data.reason_mapper import map_reason

    hist = _normalize_history(pd.read_csv(RAW / "nyc_schedule_history.csv", low_memory=False))
    severe = _severe_labels(hist)
    factors = factor_ids()

    rows = []
    for pid, g in hist.groupby("pid", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        cut = max(1, int(np.floor(n * early_frac)))
        early = g.iloc[:cut].copy()
        early["reason"] = early["reason"].fillna("").astype(str)
        early["factor_id"] = early["reason"].map(lambda t: map_reason(t, use_nlp=True) if str(t).strip() else "")
        pos = early.loc[early["variance_day"] > 0, "variance_day"]
        feat = {
            "pid": str(pid),
            "n_early_updates": int(len(early)),
            "n_total_updates": int(n),
            "early_pos_var_sum": float(pos.sum()) if len(pos) else 0.0,
            "early_pos_var_max": float(pos.max()) if len(pos) else 0.0,
            "early_pos_var_mean": float(pos.mean()) if len(pos) else 0.0,
            "early_share_positive": float((early["variance_day"] > 0).mean()),
            "n_reason_events": int((early["reason"].str.strip() != "").sum()),
            "n_unique_factors": int(early.loc[early["factor_id"] != "", "factor_id"].nunique()),
            "phase": str(early["phase"].iloc[-1] if len(early) else ""),
            "managing_agency": str(early["managing_agency"].iloc[-1] if len(early) else ""),
            "severe_delay": int(severe.get(pid, 0)),
        }
        for f in factors:
            feat[f"fac_{f}"] = int((early["factor_id"] == f).any())
        rows.append(feat)

    df = pd.DataFrame(rows)

    # Join capital-project covariates when available (not used for label)
    proj_path = RAW / "nyc_capital_projects.csv"
    if proj_path.exists():
        proj = pd.read_csv(proj_path, low_memory=False)
        plower = {c.lower().strip().replace(" ", "_"): c for c in proj.columns}
        prename = {}
        for canon, cands in {
            "pid": ["pid"],
            "category": ["category"],
            "borough": ["borough"],
            "budget_forecast": ["budget_forecast"],
            "total_budget_changes": ["total_budget_changes"],
        }.items():
            for cand in cands:
                if cand in plower:
                    prename[plower[cand]] = canon
                    break
        proj = proj.rename(columns=prename)
        if "pid" in proj.columns:
            proj["pid"] = proj["pid"].astype(str)
            # latest row per pid
            proj = proj.drop_duplicates("pid", keep="last")
            keep = [c for c in ["pid", "category", "borough", "budget_forecast", "total_budget_changes"] if c in proj.columns]
            df = df.merge(proj[keep], on="pid", how="left")
            if "budget_forecast" in df.columns:
                df["budget_forecast"] = (
                    df["budget_forecast"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
                )
                df["budget_forecast"] = pd.to_numeric(df["budget_forecast"], errors="coerce").fillna(0.0)
            if "total_budget_changes" in df.columns:
                df["total_budget_changes"] = (
                    df["total_budget_changes"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
                )
                df["total_budget_changes"] = pd.to_numeric(df["total_budget_changes"], errors="coerce").fillna(0.0)

    # Stable split by pid
    rng = np.random.default_rng(seed)
    pids = df["pid"].to_numpy()
    idx = np.arange(len(pids))
    rng.shuffle(idx)
    n = len(idx)
    n_train, n_val = int(0.70 * n), int(0.15 * n)
    split = np.array(["test"] * n, dtype=object)
    split[idx[:n_train]] = "train"
    split[idx[n_train : n_train + n_val]] = "val"
    df["split"] = split

    out = PROC / "early_warning_table.csv"
    df.to_csv(out, index=False)
    return df
