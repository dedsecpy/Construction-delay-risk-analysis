"""Build per-PID sequential episodes from NYC Open Data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.reason_mapper import factor_ids, map_reason

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

MIN_HISTORY_ROWS = 20_000
MIN_DISTINCT_PIDS = 3_000


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip().replace(" ", "_").replace("(", "").replace(")", ""): c for c in df.columns}
    rename = {}
    mapping = {
        "pid": ["pid"],
        "reporting_period": ["reporting_period"],
        "data_date": ["data_date"],
        "variance_day": ["variance_day", "variance_day"],
        "reason": ["reason_for_forecast_completion_change"],
        "phase": ["current_phase"],
        "managing_agency": ["managing_agency"],
        "project_name": ["agency_project_name"],
    }
    lower_cols = {c.lower().strip().replace(" ", "_").replace("(", "").replace(")", ""): c for c in df.columns}
    for canon, candidates in mapping.items():
        for cand in candidates:
            key = cand.lower()
            if key in lower_cols:
                rename[lower_cols[key]] = canon
                break
    out = df.rename(columns=rename).copy()
    required = ["pid", "reporting_period", "variance_day"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"History CSV missing columns {missing}; got {list(out.columns)}")
    if "reason" not in out.columns:
        out["reason"] = ""
    out["pid"] = out["pid"].astype(str)
    out["variance_day"] = pd.to_numeric(out["variance_day"], errors="coerce").fillna(0.0)
    out["reporting_period"] = pd.to_numeric(out["reporting_period"], errors="coerce")
    out = out.sort_values(["pid", "reporting_period", "data_date"] if "data_date" in out.columns else ["pid", "reporting_period"])
    return out


def _severe_labels(history: pd.DataFrame) -> pd.Series:
    """Top-quartile cumulative positive variance among PIDs with >=2 rows."""
    cum = (
        history.assign(pos_var=history["variance_day"].clip(lower=0))
        .groupby("pid")["pos_var"]
        .sum()
    )
    counts = history.groupby("pid").size()
    eligible = cum[counts >= 2]
    if eligible.empty:
        return pd.Series(dtype=int)
    thr = eligible.quantile(0.75)
    severe = (eligible >= thr).astype(int)
    # PIDs with <2 rows: not severe by default
    all_pids = cum.index
    return severe.reindex(all_pids, fill_value=0).astype(int)


def build_episodes(seed: int = 42, use_nlp: bool = True) -> pd.DataFrame:
    hist_path = RAW / "nyc_schedule_history.csv"
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing {hist_path}; run: python -m src.data.download_datasets")

    raw = pd.read_csv(hist_path, low_memory=False)
    if len(raw) < MIN_HISTORY_ROWS:
        raise AssertionError(f"Expected >= {MIN_HISTORY_ROWS} history rows, got {len(raw)}")
    history = _normalize_history(raw)
    n_pids = history["pid"].nunique()
    if n_pids < MIN_DISTINCT_PIDS:
        raise AssertionError(f"Expected >= {MIN_DISTINCT_PIDS} distinct PIDs, got {n_pids}")

    history["factor_id"] = history["reason"].map(lambda t: map_reason(t, use_nlp=use_nlp))
    severe = _severe_labels(history)

    rows: list[dict] = []
    for pid, g in history.groupby("pid", sort=False):
        g = g.reset_index(drop=True)
        cum = 0.0
        step_i = 0
        for _, r in g.iterrows():
            cum += max(float(r["variance_day"]), 0.0)
            raw_reason = r.get("reason", "")
            if raw_reason is None or (isinstance(raw_reason, float) and pd.isna(raw_reason)):
                reason = ""
            else:
                reason = str(raw_reason).strip()
                if reason.lower() in {"nan", "none", "null"}:
                    reason = ""
            # Empty reasons carry schedule variance for labels, but are NOT factor evidence
            if not reason:
                continue
            rows.append(
                {
                    "pid": pid,
                    "step": int(step_i),
                    "reporting_period": r.get("reporting_period"),
                    "reason": reason,
                    "factor_id": r["factor_id"],
                    "observed": 1,
                    "variance_day": float(r["variance_day"]),
                    "cumulative_variance": cum,
                    "severe_delay": int(severe.get(pid, 0)),
                    "phase": r.get("phase", ""),
                    "managing_agency": r.get("managing_agency", ""),
                    "project_name": r.get("project_name", ""),
                }
            )
            step_i += 1
    episodes = pd.DataFrame(rows)

    # Ensure every PID appears at least once for split/label integrity
    all_pids = history["pid"].astype(str).unique()
    present_pids = set(episodes["pid"].astype(str).unique()) if len(episodes) else set()
    missing = [p for p in all_pids if p not in present_pids]
    if missing:
        extra = []
        for p in missing:
            extra.append(
                {
                    "pid": p,
                    "step": 0,
                    "reporting_period": np.nan,
                    "reason": "",
                    "factor_id": "other_admin",
                    "observed": 0,
                    "variance_day": 0.0,
                    "cumulative_variance": float(
                        history.loc[history["pid"].astype(str) == p, "variance_day"].clip(lower=0).sum()
                    ),
                    "severe_delay": int(severe.get(p, 0)),
                    "phase": "",
                    "managing_agency": "",
                    "project_name": "",
                }
            )
        episodes = pd.concat([episodes, pd.DataFrame(extra)], ignore_index=True)

    # PID-level split 70/15/15
    rng = np.random.default_rng(seed)
    pids = np.array(episodes["pid"].astype(str).unique())
    rng.shuffle(pids)
    n = len(pids)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    split_map = {}
    for i, pid in enumerate(pids):
        if i < n_train:
            split_map[pid] = "train"
        elif i < n_train + n_val:
            split_map[pid] = "val"
        else:
            split_map[pid] = "test"
    episodes["split"] = episodes["pid"].map(split_map)

    PROC.mkdir(parents=True, exist_ok=True)
    out_csv = PROC / "episodes.csv"
    episodes.to_csv(out_csv, index=False)
    splits = {
        "n_rows_history": int(len(history)),
        "n_pids": int(n_pids),
        "n_episode_rows": int(len(episodes)),
        "severe_rate": float(episodes.drop_duplicates("pid")["severe_delay"].mean()),
        "factors": factor_ids(),
        "counts": {k: int(v) for k, v in episodes.drop_duplicates("pid")["split"].value_counts().items()},
    }
    (PROC / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(json.dumps(splits, indent=2))
    print(f"[ok] wrote {out_csv}")
    return episodes


def main() -> None:
    build_episodes()


if __name__ == "__main__":
    main()
