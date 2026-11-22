"""Rebuild episodes with updated reason→factor mapping and write count summary."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.nyc_ingest import build_episodes
from src.data.reason_mapper import factor_ids, load_patterns, reset_nlp_cache


def main() -> None:
    reset_nlp_cache()
    load_patterns.cache_clear()
    # Deterministic ontology alignment for canonical NYC phrases
    ep = build_episodes(seed=42, use_nlp=False)
    factors = factor_ids()
    # counts on real reason text only (exclude PID placeholders with blank reasons)
    obs = ep[ep["reason"].fillna("").astype(str).str.strip().str.lower().isin({"", "nan", "none"}) == False]
    rows = obs.groupby("factor_id").size().reindex(factors, fill_value=0).astype(int)
    pids = obs.groupby("factor_id")["pid"].nunique().reindex(factors, fill_value=0).astype(int)
    summary = pd.DataFrame({"rows": rows, "projects": pids})
    out = ROOT / "data" / "processed" / "factor_category_counts.csv"
    summary.to_csv(out)
    print("Non-empty reason rows by category:")
    print(summary.to_string())
    print(f"\nEpisode rows total: {len(ep)} (includes blank placeholders for PIDs with no reasons)")
    print(f"Wrote {ROOT / 'data' / 'processed' / 'episodes.csv'}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
