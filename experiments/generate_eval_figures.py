"""Generate evaluation figure pack for DRIN (held-out TEST PIDs; no fabricated metrics)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bn.inference import DelayRiskBN
from src.bn.learn_cpts import load_learned_cpts
from src.data.reason_mapper import factor_ids, map_reason
from src.eval.calibrate import load_thresholds
from src.eval.episode_replay import replay_episode
from src.eval.metrics import summarize, traces_to_frame
from src.policy.consensus_voi import ConsensusVoIPolicy
from src.policy.disagreement_voi import DisagreementVoIPolicy
from src.policy.fixed_ranking import FixedRankingPolicy
from src.policy.static_bn import StaticBNPolicy
from src.policy.utility import FA_COST, MISS_COST, expected_alert_utility

PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
RAW = ROOT / "data" / "raw"

# Visual system — Turner-adjacent ops palette (no purple/cream AI defaults)
BLUE = "#0B5DD0"
RED = "#FF4026"
INK = "#17171B"
MUTED = "#5C6370"
GRID = "#E6E8EC"
OK = "#1F7A4C"
LIT_A = "#8A94A6"
LIT_B = "#C45C26"


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def _save(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] {path}")
    return path


def load_champion_test():
    bundle = joblib.load(PROC / "champion_model.joblib")
    meta = json.loads((PROC / "champion_meta.json").read_text(encoding="utf-8"))
    df = pd.read_csv(PROC / "early_warning_table.csv")
    test = df[df["split"] == "test"].copy()
    feats = bundle["num_cols"] + bundle["cat_cols"]
    model = bundle["model"]
    thr = float(bundle["threshold"])
    y = test["severe_delay"].astype(int).to_numpy()
    p = model.predict_proba(test[feats])[:, 1]
    pred = (p >= thr).astype(int)
    return test, y, p, pred, thr, meta


def bootstrap_ci(y, pred=None, p=None, metric="accuracy", n_boot=400, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y[idx]
        if metric == "accuracy":
            vals.append(accuracy_score(yt, pred[idx]))
        elif metric == "recall":
            vals.append(recall_score(yt, pred[idx], zero_division=0))
        elif metric == "precision":
            vals.append(precision_score(yt, pred[idx], zero_division=0))
        elif metric == "f1":
            vals.append(f1_score(yt, pred[idx], zero_division=0))
        elif metric == "roc_auc":
            if len(np.unique(yt)) < 2:
                continue
            vals.append(roc_auc_score(yt, p[idx]))
        elif metric == "brier":
            vals.append(brier_score_loss(yt, p[idx]))
        elif metric == "balanced_accuracy":
            vals.append(balanced_accuracy_score(yt, pred[idx]))
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def fig_literature_beat(meta):
    our = float(meta["test_metrics"]["accuracy"])
    rows = [
        ("Khodabakhshian\npre-GAN", 0.72),
        ("Khodabakhshian\npost-GAN", 0.85),
        ("Tagharobi\nexternal", 0.768),
        ("Tagharobi\nmodel-3", 0.89),
        ("DRIN champion\n(TEST)", our),
    ]
    labels, vals = zip(*rows)
    colors = [LIT_A, LIT_A, LIT_A, LIT_B, BLUE]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bars = ax.bar(labels, vals, color=colors, width=0.72, edgecolor="white")
    ax.axhline(0.85, color=LIT_A, ls="--", lw=1.2, label="Khodabakhshian 85%")
    ax.axhline(0.89, color=LIT_B, ls="--", lw=1.2, label="Tagharobi 89%")
    ax.axhline(our, color=BLUE, ls=":", lw=1.4, label=f"DRIN {our:.3f}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0.55, 1.02)
    ax.set_ylabel("Accuracy")
    ax.set_title("Literature beat — severe-delay accuracy on held-out TEST PIDs")
    ax.legend(loc="lower right")
    ax.set_xlabel("")
    return _save(fig, "01_literature_accuracy_beat.png")


def fig_classifier_bars(y, pred, p):
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "Balanced\nacc.": balanced_accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "Specificity": tn / max(tn + fp, 1),
        "FPR": fp / max(fp + tn, 1),
        "F1": f1_score(y, pred, zero_division=0),
        "F2": fbeta_score(y, pred, beta=2, zero_division=0),
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "Brier↓": brier_score_loss(y, p),
        "Log-loss↓": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
    }
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    names = list(metrics.keys())
    vals = [metrics[k] for k in names]
    colors = [RED if "↓" in n or n == "FPR" else BLUE for n in names]
    bars = ax.bar(names, vals, color=colors, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("DRIN champion — standard delay-classifier metrics (TEST)")
    return _save(fig, "02_classifier_metric_bars.png"), metrics


def fig_roc_pr(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    base = float(y.mean())

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    axes[0].plot(fpr, tpr, color=BLUE, lw=2.4, label=f"DRIN ROC-AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate (recall)")
    axes[0].set_title("ROC curve (TEST)")
    axes[0].legend(loc="lower right")

    axes[1].plot(rec, prec, color=RED, lw=2.4, label=f"DRIN PR-AUC = {ap:.3f}")
    axes[1].axhline(base, color=MUTED, ls="--", lw=1, label=f"Base rate = {base:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall curve (TEST)")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    return _save(fig, "03_roc_pr_curves.png")


def fig_confusion(y, pred, thr):
    acc = accuracy_score(y, pred)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ConfusionMatrixDisplay.from_predictions(
        y,
        pred,
        display_labels=["Not severe", "Severe"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Confusion matrix (TEST)\nAccuracy = {acc:.3f} · τ = {thr:.2f}")
    return _save(fig, "04_confusion_matrix.png")


def fig_calibration(y, p):
    prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    # ECE (equal-width)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < 9 else p <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    brier = brier_score_loss(y, p)
    ll = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", color=BLUE, lw=2.2, ms=7, label="DRIN (quantile bins)")
    ax.set_xlabel("Predicted P(severe)")
    ax.set_ylabel("Observed severe rate")
    ax.set_title(f"Calibration curve (TEST)\nBrier={brier:.3f} · Log-loss={ll:.3f} · ECE={ece:.3f}")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return _save(fig, "05_calibration_curve.png"), {"brier": brier, "log_loss": ll, "ece": ece}


def fig_decision_curve(y, p):
    thresholds = np.linspace(0.05, 0.95, 37)
    n = len(y)
    nb_model, nb_all = [], []
    for pt in thresholds:
        pred = (p >= pt).astype(int)
        tp = ((pred == 1) & (y == 1)).sum() / n
        fp = ((pred == 1) & (y == 0)).sum() / n
        nb_model.append(tp - fp * (pt / (1 - pt)))
        prevalence = y.mean()
        nb_all.append(prevalence - (1 - prevalence) * (pt / (1 - pt)))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(thresholds, nb_model, color=BLUE, lw=2.4, label="DRIN champion")
    ax.plot(thresholds, nb_all, color=MUTED, ls="--", lw=1.4, label="Alert-all")
    ax.axhline(0, color=INK, lw=1, label="Alert-none")
    ax.set_xlabel("Decision threshold (pt)")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision-curve analysis — operational value beyond AUC (TEST)")
    ax.legend(loc="upper right")
    # Focus on the decision-relevant band (alert-all collapses at high pt)
    ymax = max(0.05, max(nb_model) * 1.25)
    ax.set_ylim(-0.05, ymax)
    return _save(fig, "06_decision_curve_net_benefit.png")


def fig_bootstrap_and_split(y, pred, p, meta):
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "brier"]
    rows = []
    for m in metrics:
        mean, lo, hi = bootstrap_ci(y, pred=pred, p=p, metric=m)
        rows.append((m, mean, lo, hi))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ys = np.arange(len(rows))
    means = [r[1] for r in rows]
    los = [r[1] - r[2] for r in rows]
    his = [r[3] - r[1] for r in rows]
    ax.barh(ys, means, xerr=[los, his], color=BLUE, alpha=0.9, capsize=4, ecolor=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score (95% bootstrap CI)")
    ax.set_title("TEST metrics with 95% bootstrap CIs (n=400)")

    ax2 = axes[1]
    splits = [
        ("Train PIDs", meta["n_train"], BLUE),
        ("Val PIDs", meta["n_val"], MUTED),
        ("Test PIDs", meta["n_test"], RED),
    ]
    ax2.bar([s[0] for s in splits], [s[1] for s in splits], color=[s[2] for s in splits])
    for i, s in enumerate(splits):
        ax2.text(i, s[1] + 40, str(s[1]), ha="center", fontweight="bold")
    ax2.set_ylabel("Projects (PID)")
    ax2.set_title("PID-level held-out split (70 / 15 / 15)")
    ax2.set_ylim(0, max(s[1] for s in splits) * 1.18)
    fig.tight_layout()
    return _save(fig, "07_bootstrap_ci_and_pid_split.png"), rows


def fig_literature_table(meta):
    rows = meta["literature_comparison"]
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.axis("off")
    cell = [
        [
            r["literature_metric"].replace("_", " "),
            f"{r['literature_value']:.3f}",
            f"{r['our_test_accuracy']:.3f}",
            f"{r['our_test_auc']:.3f}",
            "YES" if r["beat"] else "NO",
            f"{r['margin_accuracy']:+.3f}",
        ]
        for r in rows
    ]
    table = ax.table(
        cellText=cell,
        colLabels=["Literature target", "Lit. acc.", "DRIN acc.", "DRIN AUC", "Beat?", "Δ acc."],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.15, 1.55)
    for (r, c), cell_obj in table.get_celld().items():
        cell_obj.set_edgecolor(GRID)
        if r == 0:
            cell_obj.set_facecolor(BLUE)
            cell_obj.set_text_props(color="white", fontweight="bold")
        elif c == 4:
            cell_obj.set_text_props(color=OK if cell[r - 1][4] == "YES" else RED, fontweight="bold")
    ax.set_title("Literature comparison table (TEST)", pad=18, fontweight="bold")
    return _save(fig, "08_literature_comparison_table.png")


def run_policy_scoreboard():
    episodes = pd.read_csv(PROC / "episodes.csv")
    learned = load_learned_cpts()
    thr_payload = load_thresholds()
    thr_map = {k: float(v["threshold"]) for k, v in thr_payload.get("policies", thr_payload).items()}
    # load_thresholds shape from calibrated_thresholds.json
    if "policies" in thr_payload:
        thr_map = {k: float(v["threshold"]) for k, v in thr_payload["policies"].items()}

    factors = factor_ids()
    order = sorted(factors, key=lambda f: learned["consensus_priors"][f], reverse=True)
    policies = [
        FixedRankingPolicy(order),
        StaticBNPolicy(order),
        ConsensusVoIPolicy(lookahead=1),
        DisagreementVoIPolicy(lookahead=1),
    ]
    budgets = (1, 2, 3, 4)
    test_pids = episodes.loc[episodes["split"] == "test", "pid"].astype(str).unique()
    traces = []
    for i, pid in enumerate(test_pids, start=1):
        if i == 1 or i % 50 == 0 or i == len(test_pids):
            print(f"[policy TEST] {i}/{len(test_pids)}", flush=True)
        ep = episodes[episodes["pid"].astype(str) == pid]
        for B in budgets:
            for pol in policies:
                priors = (
                    learned["disagreement_priors"]
                    if pol.name == "disagreement_voi"
                    else learned["consensus_priors"]
                )
                bn = DelayRiskBN(
                    factor_priors=priors,
                    p_cause=learned["p_cause"],
                    leak=learned["leak"],
                )
                thr = float(thr_map.get(pol.name, 0.57))
                traces.append(replay_episode(pol, bn, ep, B, factors, threshold=thr))
    df = traces_to_frame(traces)
    board = summarize(df)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "eval_traces_test.csv", index=False)
    board.to_csv(RESULTS / "scoreboard_improved.csv", index=False)
    return df, board


def fig_budget_metrics(board: pd.DataFrame):
    policies = ["fixed", "static_bn", "consensus_voi", "disagreement_voi"]
    labels = {
        "fixed": "Fixed",
        "static_bn": "Static BN",
        "consensus_voi": "Consensus VoI",
        "disagreement_voi": "Disagreement VoI",
    }
    colors = {
        "fixed": MUTED,
        "static_bn": LIT_A,
        "consensus_voi": BLUE,
        "disagreement_voi": RED,
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True)
    metrics = [("recall", "Recall@B"), ("precision", "Precision@B"), ("roc_auc", "AUC@B")]
    for ax, (col, title) in zip(axes, metrics):
        for pol in policies:
            g = board[board["policy"] == pol].sort_values("budget")
            ax.plot(g["budget"], g[col], "o-", color=colors[pol], lw=2.1, ms=6, label=labels[pol])
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("Inspection budget B")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Budgeted sequential inspection — TEST PID scoreboard", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "09_recall_precision_auc_at_B.png")


def fig_inspections_to_detection(traces: pd.DataFrame):
    # Among severe projects that alerted, mean steps_to_alert by policy at B=4 (full budget)
    g = traces[(traces["budget"] == 4) & (traces["severe_delay"] == 1) & traces["steps_to_alert"].notna()]
    if g.empty:
        g = traces[(traces["budget"] == 4) & (traces["severe_delay"] == 1)]
        # fallback: use n_inspections when alerted
        g = traces[(traces["budget"] == 4) & (traces["severe_delay"] == 1) & (traces["alerted"] == 1)]
        vals = g.groupby("policy")["n_inspections"].mean()
        ylabel = "Mean inspections until alert (severe & alerted)"
    else:
        vals = g.groupby("policy")["steps_to_alert"].mean()
        ylabel = "Mean inspections-to-first-detection (severe)"

    order = ["fixed", "static_bn", "consensus_voi", "disagreement_voi"]
    labels = ["Fixed", "Static BN", "Consensus VoI", "Disagr. VoI"]
    colors = [MUTED, LIT_A, BLUE, RED]
    data = [float(vals.get(p, np.nan)) for p in order]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, data, color=colors, edgecolor="white")
    for b, v in zip(bars, data):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title("VoI efficiency — inspections to first detection (TEST, B=4)")
    return _save(fig, "10_inspections_to_first_detection.png"), data


def fig_lead_time(test: pd.DataFrame, y, pred):
    # Lead time proxy: remaining reporting updates after early-warning window
    # only for true severe projects that the champion flags
    severe = test[y == 1].copy()
    severe["pred"] = pred[y == 1]
    severe["lead_updates"] = (severe["n_total_updates"] - severe["n_early_updates"]).clip(lower=0)
    caught = severe[severe["pred"] == 1]["lead_updates"]
    missed = severe[severe["pred"] == 0]["lead_updates"]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bins = np.arange(0, max(int(severe["lead_updates"].max()) + 2, 6))
    ax.hist(caught, bins=bins, alpha=0.85, color=BLUE, label=f"Caught (n={len(caught)})", edgecolor="white")
    ax.hist(missed, bins=bins, alpha=0.55, color=RED, label=f"Missed (n={len(missed)})", edgecolor="white")
    ax.axvline(caught.mean() if len(caught) else 0, color=BLUE, ls="--", lw=1.6, label=f"Mean lead (caught)={caught.mean():.2f}")
    ax.set_xlabel("Lead time (remaining updates after early window)")
    ax.set_ylabel("Severe projects (TEST)")
    ax.set_title("Lead time to alert — early-warning horizon on severe TEST PIDs")
    ax.legend()
    return _save(fig, "11_lead_time_to_alert.png"), float(caught.mean() if len(caught) else np.nan)


def fig_nlp_factor_f1():
    """NLP-only F1 vs weak regex labels on VAL (honest: does not use map_reason which prefers regex)."""
    from sklearn.metrics import classification_report
    from src.data.nlp_reason_classifier import _clean, build_weak_labels, load_nlp_classifier
    from src.data.reason_mapper import map_reason_regex

    hist = pd.read_csv(RAW / "nyc_schedule_history.csv", low_memory=False)
    reason_col = next(c for c in hist.columns if "reason" in c.lower())
    pid_col = next((c for c in hist.columns if c.lower() in {"project_id", "pid"}), hist.columns[0])
    ew = pd.read_csv(PROC / "early_warning_table.csv", usecols=["pid", "split"])
    split_map = {str(r.pid): str(r.split) for r in ew.itertuples(index=False)}

    hist = hist.copy()
    hist["pid"] = hist[pid_col].astype(str)
    hist["reason"] = hist[reason_col].astype(str)
    hist = hist[~hist["reason"].str.strip().str.lower().isin(["", "nan", "none", "null"])]
    hist["split"] = hist["pid"].map(split_map)
    val = hist.loc[hist["split"] == "val", "reason"].fillna("").astype(str)
    rows = []
    for text in val:
        t = _clean(text)
        if not t:
            continue
        gold = map_reason_regex(t)
        if gold == "other_admin":
            continue
        rows.append({"text": t, "factor_id": gold})
    labeled = pd.DataFrame(rows)
    mdl = load_nlp_classifier()
    if mdl is None or labeled.empty:
        raise RuntimeError("NLP model or VAL labels missing")

    # NLP-only prediction (bypass regex preference in map_reason)
    y_true = labeled["factor_id"].tolist()
    classes = list(mdl.classes_)
    y_pred = []
    for t in labeled["text"].tolist():
        proba = mdl.predict_proba([t])[0]
        y_pred.append(str(classes[int(np.argmax(proba))]))

    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    macro = float(report["macro avg"]["f1-score"])
    weighted = float(report["weighted avg"]["f1-score"])
    per = [(lab, report[lab]["f1-score"], int(report[lab]["support"])) for lab in labels if lab in report]

    full_pred = [map_reason(t) for t in labeled["text"].tolist()]
    full_f1 = float(f1_score(y_true, full_pred, average="macro", labels=labels, zero_division=0))
    full_wf1 = float(f1_score(y_true, full_pred, average="weighted", labels=labels, zero_division=0))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), sharey=True)
    labs = [p[0].replace("_", "\n") for p in per]
    f1s = [p[1] for p in per]
    supports = [p[2] for p in per]
    bars = axes[0].bar(labs, f1s, color=BLUE, edgecolor="white")
    for b, v, s in zip(bars, f1s, supports):
        axes[0].text(b.get_x() + b.get_width() / 2, min(v + 0.03, 1.12), f"{v:.2f}\n(n={s})", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylim(0, 1.25)
    axes[0].set_ylabel("F1")
    axes[0].set_title(f"NLP-only vs regex labels\nmacro={macro:.3f} · wtd={weighted:.3f}")

    # Full mapper (regex-first) — operational evidence path
    full_report = classification_report(y_true, full_pred, labels=labels, output_dict=True, zero_division=0)
    full_per = [(lab, full_report[lab]["f1-score"], int(full_report[lab]["support"])) for lab in labels if lab in full_report]
    bars = axes[1].bar([p[0].replace("_", "\n") for p in full_per], [p[1] for p in full_per], color=OK, edgecolor="white")
    for b, v, s in zip(bars, [p[1] for p in full_per], [p[2] for p in full_per]):
        axes[1].text(b.get_x() + b.get_width() / 2, min(v + 0.03, 1.12), f"{v:.2f}\n(n={s})", ha="center", va="bottom", fontsize=8)
    axes[1].set_title(f"Full mapper (regex-first) vs regex labels\nmacro={full_f1:.3f} · wtd={full_wf1:.3f}")

    fig.suptitle(f"Reason→factor mapping quality (VAL rows, n={len(y_true)})", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "12_reason_factor_mapping_f1.png"), {
        "macro_f1": macro,
        "weighted_f1": weighted,
        "full_mapper_macro_f1": full_f1,
        "full_mapper_weighted_f1": full_wf1,
        "n": len(y_true),
        "n_unique_texts": int(labeled["text"].nunique()),
    }


def fig_expected_utility(traces: pd.DataFrame):
    # Realized decision loss on TEST at B=2: MISS_COST * FN + FA_COST * FP (lower better)
    # Also mean expected_alert_utility of final p_severe
    b2 = traces[traces["budget"] == 2].copy()
    rows = []
    for pol, g in b2.groupby("policy"):
        y = g["severe_delay"].to_numpy()
        a = g["alerted"].to_numpy().astype(int)
        fp = ((a == 1) & (y == 0)).sum()
        fn = ((a == 0) & (y == 1)).sum()
        realized = (MISS_COST * fn + FA_COST * fp) / len(g)
        exp_u = float(np.mean([expected_alert_utility(float(p)) for p in g["p_severe"]]))
        rows.append((pol, realized, exp_u))

    order = ["fixed", "static_bn", "consensus_voi", "disagreement_voi"]
    labels = ["Fixed", "Static BN", "Consensus VoI", "Disagr. VoI"]
    realized = [next(r[1] for r in rows if r[0] == p) for p in order]
    exp_u = [next(r[2] for r in rows if r[0] == p) for p in order]
    colors = [MUTED, LIT_A, BLUE, RED]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    bars = axes[0].bar(labels, realized, color=colors)
    for b, v in zip(bars, realized):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    axes[0].set_ylabel(f"Mean realized loss (MISS={MISS_COST}, FA={FA_COST})")
    axes[0].set_title("Realized alert loss @ B=2 (TEST) ↓ better")

    bars = axes[1].bar(labels, exp_u, color=colors)
    for b, v in zip(bars, exp_u):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.005 if v >= 0 else v - 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    axes[1].set_ylabel("Mean expected alert utility")
    axes[1].set_title("Expected utility of final P(severe) @ B=2")
    fig.suptitle("VoI policy decision quality", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "13_expected_utility_loss.png"), rows


def fig_dashboard_summary(meta, clf_metrics, ece, nlp, board):
    """One-page executive scorecard."""
    tm = meta["test_metrics"]
    fig = plt.figure(figsize=(12.5, 7.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.03, 0.93, "DRIN evaluation scorecard", fontsize=20, fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(
        0.03,
        0.88,
        "Held-out TEST PIDs · champion early-warning + budgeted BN/VoI inspection",
        fontsize=11,
        color=MUTED,
        transform=ax.transAxes,
    )

    cards = [
        ("Accuracy", f"{tm['accuracy']:.3f}", "vs lit. 0.85 / 0.89"),
        ("ROC-AUC", f"{tm['roc_auc']:.3f}", "threshold-free"),
        ("F1 / F2", f"{tm['f1']:.3f}", f"F2 via bars pack"),
        ("Brier", f"{tm['brier']:.3f}", f"ECE={ece:.3f}"),
        ("PR-AUC", f"{clf_metrics['PR-AUC']:.3f}", "imbalance-safe"),
        ("Balanced acc.", f"{tm['balanced_accuracy']:.3f}", "class-balanced"),
        ("NLP macro-F1", f"{nlp['macro_f1']:.3f}", f"n={nlp['n']} VAL"),
        ("Recall@B=2", f"{float(board[board.budget==2].iloc[0].recall):.3f}", "all policies tied*"),
    ]

    # * note if policies equal
    b2 = board[board["budget"] == 2]
    if b2["recall"].nunique() == 1:
        cards[-1] = ("Recall@B=2", f"{float(b2.iloc[0].recall):.3f}", "policies tied on scoreboard")

    for i, (title, value, sub) in enumerate(cards):
        x = 0.04 + (i % 4) * 0.24
        y = 0.55 if i < 4 else 0.22
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                0.21,
                0.26,
                boxstyle="round,pad=0.015,rounding_size=0.02",
                transform=ax.transAxes,
                facecolor="#F4F6F8",
                edgecolor=GRID,
                linewidth=1.2,
            )
        )
        ax.text(x + 0.02, y + 0.18, title, fontsize=11, color=MUTED, transform=ax.transAxes)
        ax.text(x + 0.02, y + 0.08, value, fontsize=22, fontweight="bold", color=BLUE, transform=ax.transAxes)
        ax.text(x + 0.02, y + 0.03, sub, fontsize=9, color=MUTED, transform=ax.transAxes)

    ax.text(
        0.03,
        0.08,
        "Sources: champion_model.joblib + episodes.csv TEST replay · figures in results/figures/",
        fontsize=9,
        color=MUTED,
        transform=ax.transAxes,
    )
    return _save(fig, "00_scorecard_overview.png")


def main():
    _style()
    FIG.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    test, y, p, pred, thr, meta = load_champion_test()
    print(f"TEST n={len(y)} severe_rate={y.mean():.3f} thr={thr:.3f}")

    fig_literature_beat(meta)
    _, clf_metrics = fig_classifier_bars(y, pred, p)
    fig_roc_pr(y, p)
    fig_confusion(y, pred, thr)
    _, cal = fig_calibration(y, p)
    fig_decision_curve(y, p)
    _, boot_rows = fig_bootstrap_and_split(y, pred, p, meta)
    fig_literature_table(meta)
    lead_mean = None
    _, lead_mean = fig_lead_time(test, y, pred)

    print("Running TEST policy scoreboard (this can take a few minutes)...")
    traces, board = run_policy_scoreboard()
    fig_budget_metrics(board)
    fig_inspections_to_detection(traces)
    _, nlp = fig_nlp_factor_f1()
    fig_expected_utility(traces)
    fig_dashboard_summary(meta, clf_metrics, cal["ece"], nlp, board)

    summary = {
        "test_n": int(len(y)),
        "threshold": thr,
        "champion_test_metrics": meta["test_metrics"],
        "derived": {
            "pr_auc": clf_metrics["PR-AUC"],
            "f2": clf_metrics["F2"],
            "ece": cal["ece"],
            "lead_time_mean_caught": lead_mean,
            "nlp": nlp,
        },
        "bootstrap_95ci": [
            {"metric": m, "mean": mean, "lo": lo, "hi": hi} for m, mean, lo, hi in boot_rows
        ],
        "scoreboard_b2": board[board["budget"] == 2].to_dict(orient="records"),
        "figures_dir": str(FIG),
    }
    (RESULTS / "eval_figures_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE", FIG)


if __name__ == "__main__":
    main()
