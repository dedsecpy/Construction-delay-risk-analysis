"""Download NYC Open Data + UK GMPP CSVs into data/raw."""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

DATASETS = {
    "nyc_schedule_history.csv": "https://data.cityofnewyork.us/api/v3/views/95tx-snak/export.csv?accessType=DOWNLOAD",
    "nyc_capital_projects.csv": "https://data.cityofnewyork.us/api/v3/views/n7gv-k5yt/export.csv?accessType=DOWNLOAD",
    "uk_gmpp_2025_26.csv": "https://assets.publishing.service.gov.uk/media/6a4fb28da6586e258d371bfd/NISTA_Major_Projects_Annual_Report_2025-2026_Data.csv",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(name: str, url: str, force: bool = False) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / name
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"[skip] {name} already present ({dest.stat().st_size:,} bytes)")
        return dest
    print(f"[download] {name}")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    print(f"[ok] {name}: {dest.stat().st_size:,} bytes sha256={_sha256(dest)[:12]}")
    return dest


def main(force: bool = False) -> None:
    for name, url in DATASETS.items():
        download_one(name, url, force=force)


if __name__ == "__main__":
    main()
