"""Map NYC delay-reason text to the 8 locked risk factors (regex + NLP)."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REASON_MAP = ROOT / "data" / "processed" / "reason_map.csv"
FACTORS = ROOT / "data" / "processed" / "factors.csv"

_NLP_MODEL = None
_NLP_LOADED = False


@lru_cache(maxsize=1)
def load_patterns() -> list[tuple[re.Pattern[str], str]]:
    df = pd.read_csv(REASON_MAP)
    patterns: list[tuple[re.Pattern[str], str]] = []
    for _, row in df.iterrows():
        pat = str(row["pattern"]).strip()
        fid = str(row["factor_id"]).strip()
        if not pat or pat.lower() == "nan":
            continue
        patterns.append((re.compile(pat, re.IGNORECASE), fid))
    return patterns


def factor_ids() -> list[str]:
    return pd.read_csv(FACTORS)["factor_id"].tolist()


def map_reason_regex(text: str | float | None) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return "other_admin"
    s = str(text).strip()
    if not s:
        return "other_admin"
    for pattern, fid in load_patterns():
        if pattern.search(s):
            return fid
    return "other_admin"


def _get_nlp():
    global _NLP_MODEL, _NLP_LOADED
    if _NLP_LOADED:
        return _NLP_MODEL
    try:
        from src.data.nlp_reason_classifier import load_nlp_classifier

        _NLP_MODEL = load_nlp_classifier()
    except Exception:
        _NLP_MODEL = None
    _NLP_LOADED = True
    return _NLP_MODEL


def map_reason(text: str | float | None, use_nlp: bool = True) -> str:
    """Map reason text → factor_id.

    Prefer high-confidence regex hits (non-other_admin) so canonical NYC phrases
    land in the intended ontology categories; fall back to NLP, then regex.
    """
    regex_id = map_reason_regex(text)
    if regex_id != "other_admin":
        return regex_id
    if not use_nlp:
        return regex_id
    model = _get_nlp()
    if model is None:
        return regex_id
    from src.data.nlp_reason_classifier import predict_factor

    return predict_factor(text if text is not None else "", model=model)


def reset_nlp_cache() -> None:
    global _NLP_MODEL, _NLP_LOADED
    _NLP_MODEL = None
    _NLP_LOADED = False
