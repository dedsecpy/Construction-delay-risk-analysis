"""Smoke tests for literature-champion early-warning pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_early_warning_table_exists_or_builds():
    path = ROOT / "data" / "processed" / "early_warning_table.csv"
    if not path.exists():
        from src.ml.features import build_early_warning_table

        df = build_early_warning_table(early_frac=0.5, seed=42)
    else:
        df = pd.read_csv(path)
    assert len(df) >= 3000
    assert {"pid", "severe_delay", "split"}.issubset(df.columns)
    assert set(df["split"]) == {"train", "val", "test"}
    assert 0.15 <= df["severe_delay"].mean() <= 0.30


def test_literature_champion_beats_targets():
    meta = ROOT / "results" / "literature_champion.json"
    assert meta.exists(), "Run experiments/run_literature_champion.py first"
    import json

    payload = json.loads(meta.read_text(encoding="utf-8"))
    acc = payload["test_metrics"]["accuracy"]
    auc = payload["test_metrics"]["roc_auc"]
    assert acc > 0.89, f"accuracy {acc} did not beat Tagharobi 0.89"
    assert acc > 0.85, f"accuracy {acc} did not beat Khodabakhshian 0.85"
    assert auc > 0.90
    assert payload["beats_khodabakhshian_85"] is True
    assert payload["beats_tagharobi_89"] is True
