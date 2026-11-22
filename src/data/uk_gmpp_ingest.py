"""Lightweight UK GMPP loader for transfer-validation summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def load_gmpp() -> pd.DataFrame:
    path = RAW / "uk_gmpp_2025_26.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run download_datasets first")
    df = pd.read_csv(path)
    # Normalize DCA column name variants
    dca_cols = [c for c in df.columns if "delivery confidence" in c.lower() or c.lower().endswith("dca") or "IPA Delivery" in c]
    if not dca_cols:
        # try exact known header
        for c in df.columns:
            if "Confidence" in c:
                dca_cols.append(c)
                break
    if dca_cols:
        df = df.rename(columns={dca_cols[0]: "dca"})
    else:
        df["dca"] = "Unknown"
    df["high_risk"] = df["dca"].astype(str).str.contains("Red", case=False, na=False).astype(int)
    return df


def summarize() -> dict:
    df = load_gmpp()
    summary = {
        "n_projects": int(len(df)),
        "high_risk_rate": float(df["high_risk"].mean()) if len(df) else 0.0,
        "dca_counts": df["dca"].astype(str).value_counts().to_dict() if "dca" in df.columns else {},
    }
    PROC.mkdir(parents=True, exist_ok=True)
    out = PROC / "uk_gmpp_summary.json"
    import json

    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    print(summarize())


if __name__ == "__main__":
    main()
