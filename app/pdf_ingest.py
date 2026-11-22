"""Extract delay-reason candidates from uploaded PDFs and map to factors."""

from __future__ import annotations

import re
from io import BytesIO

from pypdf import PdfReader

from src.data.reason_mapper import factor_ids, map_reason

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;\n])\s+|\n+")


def extract_pdf_text(data: bytes, max_pages: int = 20) -> str:
    reader = PdfReader(BytesIO(data))
    chunks: list[str] = []
    for i, page in enumerate(reader.pages[:max_pages]):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            chunks.append(t)
    return "\n".join(chunks).strip()


def reason_candidates(text: str, limit: int = 40) -> list[str]:
    if not text.strip():
        return []
    parts = [p.strip(" -\t") for p in _SENTENCE_SPLIT.split(text) if p and p.strip()]
    # Prefer lines that look like delay/risk notes
    keywords = re.compile(
        r"delay|postpone|slip|overrun|budget|scope|design|permit|approv|procure|"
        r"contract|weather|site|utility|labor|workforce|resource|funding|grant",
        re.I,
    )
    scored: list[tuple[int, str]] = []
    for p in parts:
        if len(p) < 12 or len(p) > 400:
            continue
        score = 2 if keywords.search(p) else 0
        scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], -len(x[1])))
    # Keep keyword hits first, then fill
    picked = [s for sc, s in scored if sc > 0][:limit]
    if len(picked) < 8:
        for _, s in scored:
            if s not in picked:
                picked.append(s)
            if len(picked) >= min(limit, 16):
                break
    return picked


def map_reasons_to_factors(reasons: list[str]) -> dict:
    factors = factor_ids()
    present = {f: 0 for f in factors}
    mapped = []
    for r in reasons:
        fid = map_reason(r, use_nlp=True)
        if fid not in present:
            fid = "other_admin"
        present[fid] = 1
        mapped.append({"reason": r[:280], "factor_id": fid})
    return {
        "reasons": mapped,
        "present_factors": sorted([f for f, v in present.items() if v]),
        "factor_flags": present,
        "n_reasons": len(mapped),
    }
