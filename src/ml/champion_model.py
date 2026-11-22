"""Literature-champion severe-delay classifier (early-warning)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODEL_PATH = PROC / "champion_model.joblib"
META_PATH = PROC / "champion_meta.json"

# Published targets from literature survey
LIT_TARGETS = {
    "Khodabakhshian_2025_post_GAN_accuracy": 0.85,
    "Khodabakhshian_2025_pre_GAN_accuracy": 0.72,
    "Tagharobi_2025_model3_accuracy": 0.89,
    "Tagharobi_2025_external_accuracy": 0.768,
}


def _feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    num = [
        c
        for c in df.columns
        if c.startswith("fac_")
        or c
        in {
            "n_early_updates",
            "n_total_updates",
            "early_pos_var_sum",
            "early_pos_var_max",
            "early_pos_var_mean",
            "early_share_positive",
            "n_reason_events",
            "n_unique_factors",
            "budget_forecast",
            "total_budget_changes",
        }
    ]
    cat = [c for c in ["phase", "managing_agency", "category", "borough"] if c in df.columns]
    return num, cat


def _hgb(**kwargs) -> HistGradientBoostingClassifier:
    params = dict(
        max_depth=6,
        learning_rate=0.05,
        max_iter=500,
        l2_regularization=0.05,
        random_state=42,
    )
    params.update(kwargs)
    if "class_weight" in inspect.signature(HistGradientBoostingClassifier).parameters:
        params.setdefault("class_weight", None)
    return HistGradientBoostingClassifier(**params)


def make_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def make_stacking_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    pre = make_preprocessor(num_cols, cat_cols)
    estimators = [
        ("hgb", _hgb()),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=2,
                class_weight=None,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        (
            "lr",
            LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
        ),
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, C=1.0),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", stack)])


def make_hgb_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    return Pipeline([("pre", make_preprocessor(num_cols, cat_cols)), ("clf", _hgb())])


def _metrics(y_true, y_prob, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(((y_pred == 0) & (y_true == 0)).sum() / max(((y_true == 0).sum()), 1)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def tune_threshold(y_true, y_prob, min_recall: float = 0.55) -> float:
    """Maximize accuracy, then balanced accuracy / F1, with a soft recall floor."""
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.15, 0.85, 71),
                np.quantile(y_prob, np.linspace(0.02, 0.98, 49)),
            ]
        )
    )
    best_thr, best_key = 0.5, (-1.0, -1.0, -1.0)
    fallback_thr, fallback_key = 0.5, (-1.0, -1.0, -1.0)
    for thr in grid:
        pred = (y_prob >= thr).astype(int)
        m = _metrics(y_true, y_prob, pred)
        key = (m["accuracy"], m["balanced_accuracy"], m["f1"])
        if key > fallback_key:
            fallback_key, fallback_thr = key, float(thr)
        if m["recall"] < min_recall:
            continue
        if key > best_key:
            best_key, best_thr = key, float(thr)
    return best_thr if best_key[0] >= 0 else fallback_thr


def train_champion(df: pd.DataFrame) -> dict:
    num_cols, cat_cols = _feature_columns(df)
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()
    feats = num_cols + cat_cols

    candidates = {
        "stacking": make_stacking_pipeline(num_cols, cat_cols),
        "hgb": make_hgb_pipeline(num_cols, cat_cols),
    }

    val_y = val["severe_delay"].astype(int).to_numpy()
    best_name, best_pipe, best_thr, best_val = None, None, 0.5, {"accuracy": -1.0}

    for name, pipe in candidates.items():
        pipe.fit(train[feats], train["severe_delay"].astype(int))
        val_prob = pipe.predict_proba(val[feats])[:, 1]
        thr = tune_threshold(val_y, val_prob, min_recall=0.55)
        val_pred = (val_prob >= thr).astype(int)
        vm = _metrics(val_y, val_prob, val_pred)
        if vm["accuracy"] > best_val["accuracy"] or (
            abs(vm["accuracy"] - best_val["accuracy"]) < 1e-6 and vm["roc_auc"] > best_val.get("roc_auc", -1)
        ):
            best_name, best_pipe, best_thr, best_val = name, pipe, thr, vm

    assert best_pipe is not None
    test_prob = best_pipe.predict_proba(test[feats])[:, 1]
    test_pred = (test_prob >= best_thr).astype(int)
    test_metrics = _metrics(test["severe_delay"], test_prob, test_pred)

    comparison = []
    for name, target in LIT_TARGETS.items():
        comparison.append(
            {
                "literature_metric": name,
                "literature_value": target,
                "our_test_accuracy": test_metrics["accuracy"],
                "our_test_auc": test_metrics["roc_auc"],
                "beat": bool(test_metrics["accuracy"] > target),
                "margin_accuracy": float(test_metrics["accuracy"] - target),
            }
        )

    PROC.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_pipe,
            "model_name": best_name,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "threshold": best_thr,
        },
        MODEL_PATH,
    )

    payload = {
        "selected_model": best_name,
        "threshold": best_thr,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "base_rate_test": float(test["severe_delay"].mean()),
        "val_metrics": best_val,
        "test_metrics": test_metrics,
        "literature_comparison": comparison,
        "beats_khodabakhshian_85": bool(test_metrics["accuracy"] > 0.85),
        "beats_tagharobi_89": bool(test_metrics["accuracy"] > 0.89),
        "beats_tagharobi_external_768": bool(test_metrics["accuracy"] > 0.768),
    }
    META_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(comparison).to_csv(RESULTS / "literature_comparison.csv", index=False)
    pd.DataFrame([{"split": "val", **best_val}, {"split": "test", **test_metrics}]).to_csv(
        RESULTS / "champion_test_metrics.csv", index=False
    )
    return payload
