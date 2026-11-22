"""TF-IDF + LogisticRegression classifier for delay-reason → factor mapping."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from src.data.reason_mapper import factor_ids, map_reason_regex

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "data" / "processed" / "nlp_reason_model.joblib"
META_PATH = ROOT / "data" / "processed" / "nlp_reason_meta.json"


def _clean(text: str) -> str:
    s = str(text).upper().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_weak_labels(reasons: pd.Series) -> pd.DataFrame:
    """Label reasons with high-confidence regex; keep only non-fallback when possible."""
    rows = []
    for text in reasons.fillna("").astype(str):
        t = _clean(text)
        if not t:
            continue
        label = map_reason_regex(t)
        rows.append({"text": t, "factor_id": label})
    return pd.DataFrame(rows).drop_duplicates(subset=["text"])


def train_nlp_classifier(train_reasons: pd.Series, random_state: int = 42) -> Pipeline:
    labeled = build_weak_labels(train_reasons)
    # Prefer rows with informative labels; keep some other_admin
    informative = labeled[labeled["factor_id"] != "other_admin"]
    other = labeled[labeled["factor_id"] == "other_admin"].sample(
        n=min(len(informative), len(labeled[labeled["factor_id"] == "other_admin"])),
        random_state=random_state,
        replace=False,
    ) if (labeled["factor_id"] == "other_admin").any() else labeled.iloc[0:0]
    data = pd.concat([informative, other], ignore_index=True)
    if data["factor_id"].nunique() < 2 or len(data) < 20:
        # Fallback tiny model on all labels
        data = labeled

    pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=8000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipe.fit(data["text"].tolist(), data["factor_id"].tolist())
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    meta = {
        "n_train_texts": int(len(data)),
        "classes": sorted(data["factor_id"].unique().tolist()),
        "factors": factor_ids(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return pipe


def load_nlp_classifier() -> Pipeline | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def predict_factor(text: str, model: Pipeline | None = None, min_proba: float = 0.35) -> str:
    """Predict factor; fall back to regex if model missing or low confidence."""
    t = _clean(text)
    if not t:
        return "other_admin"
    regex_label = map_reason_regex(t)
    mdl = model if model is not None else load_nlp_classifier()
    if mdl is None:
        return regex_label
    proba = mdl.predict_proba([t])[0]
    classes = list(mdl.classes_)
    idx = int(np.argmax(proba))
    if float(proba[idx]) < min_proba:
        return regex_label
    return str(classes[idx])


def evaluate_on_texts(texts: pd.Series, model: Pipeline | None = None) -> str:
    mdl = model if model is not None else load_nlp_classifier()
    if mdl is None:
        return "no model"
    labeled = build_weak_labels(texts)
    if labeled.empty:
        return "no labels"
    pred = [predict_factor(t, mdl) for t in labeled["text"]]
    return classification_report(labeled["factor_id"], pred, zero_division=0)
