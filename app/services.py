"""Data + inspection services for DRIN (Delay Risk Inference Network)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.bn.disagreement import priors_from_episodes
from src.bn.inference import DelayRiskBN
from src.bn.learn_cpts import load_learned_cpts
from src.data.reason_mapper import factor_ids
from src.eval.episode_replay import replay_episode
from src.policy.base import Policy, PolicyState
from src.policy.consensus_voi import ConsensusVoIPolicy
from src.policy.disagreement_voi import DisagreementVoIPolicy
from src.policy.fixed_ranking import FixedRankingPolicy
from src.policy.static_bn import StaticBNPolicy
from src.policy.utility import should_alert

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
DEMO = ROOT / "data" / "demo"

FACTOR_LABELS = {
    "budget": "Budget constraints",
    "scope_design": "Scope / design",
    "grants_approvals": "Grants & approvals",
    "procurement": "Procurement",
    "construction_conditions": "Construction conditions",
    "external_dependency": "External dependency",
    "workforce_resource": "Workforce / resources",
    "other_admin": "Other administrative",
}


@lru_cache(maxsize=1)
def load_episodes() -> pd.DataFrame:
    path = PROC / "episodes.csv"
    if not path.exists():
        raise FileNotFoundError("episodes.csv missing — run ingest / improved pipeline first")
    df = pd.read_csv(path)
    df["pid"] = df["pid"].astype(str)
    return df


@lru_cache(maxsize=1)
def load_early_warning() -> pd.DataFrame | None:
    path = PROC / "early_warning_table.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["pid"] = df["pid"].astype(str)
    return df


@lru_cache(maxsize=1)
def load_champion_bundle() -> dict[str, Any] | None:
    model_path = PROC / "champion_model.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def get_factors() -> list[dict[str, str]]:
    return [{"id": f, "label": FACTOR_LABELS.get(f, f)} for f in factor_ids()]


def get_overview() -> dict[str, Any]:
    load_episodes.cache_clear()
    ep = load_episodes()
    pids = ep.groupby("pid").agg(severe_delay=("severe_delay", "max"), split=("split", "first")).reset_index()
    champion = _read_json(PROC / "champion_meta.json", {})
    lit = _read_json(RESULTS / "literature_champion.json", champion)
    scoreboard = None
    sb_path = RESULTS / "scoreboard_improved.csv"
    if sb_path.exists():
        sb = pd.read_csv(sb_path)
        scoreboard = sb[sb["budget"] == 2][
            [c for c in ["policy", "budget", "recall", "precision", "fpr", "roc_auc", "n"] if c in sb.columns]
        ].to_dict(orient="records")
    learned = _read_json(PROC / "learned_cpts.json", {})
    thresholds = _read_json(PROC / "calibrated_thresholds.json", {})
    test_m = lit.get("test_metrics") or champion.get("test_metrics") or {}
    # Factor ontology occupancy (non-empty reasons)
    counts_path = PROC / "factor_category_counts.csv"
    if counts_path.exists():
        fc = pd.read_csv(counts_path)
        if "factor_id" in fc.columns:
            factor_counts = [
                {
                    "factor": str(r["factor_id"]),
                    "label": FACTOR_LABELS.get(str(r["factor_id"]), str(r["factor_id"])),
                    "rows": int(r["rows"]),
                    "projects": int(r["projects"]),
                }
                for _, r in fc.iterrows()
            ]
        else:
            factor_counts = []
    else:
        obs = ep[ep["reason"].fillna("").astype(str).str.strip().str.lower().isin({"", "nan", "none"}) == False]
        vc = obs.groupby("factor_id").agg(rows=("factor_id", "size"), projects=("pid", "nunique")).reset_index()
        factor_counts = [
            {
                "factor": str(r["factor_id"]),
                "label": FACTOR_LABELS.get(str(r["factor_id"]), str(r["factor_id"])),
                "rows": int(r["rows"]),
                "projects": int(r["projects"]),
            }
            for _, r in vc.iterrows()
        ]

    # Severe watchlist (test split preferred)
    detail = (
        ep.groupby("pid")
        .agg(
            severe_delay=("severe_delay", "max"),
            split=("split", "first"),
            **({"project_name": ("project_name", "last")} if "project_name" in ep.columns else {}),
            **({"managing_agency": ("managing_agency", "last")} if "managing_agency" in ep.columns else {}),
            n_factors=("factor_id", "nunique"),
        )
        .reset_index()
    )
    severe = detail[detail["severe_delay"] == 1].copy()
    if "split" in severe.columns:
        severe_test = severe[severe["split"] == "test"]
        watch_src = severe_test if len(severe_test) else severe
    else:
        watch_src = severe
    watch_src = watch_src.head(12)
    severe_watchlist = []
    for _, r in watch_src.iterrows():
        severe_watchlist.append(
            {
                "pid": str(r["pid"]),
                "name": str(r.get("project_name") or f"Project {r['pid']}"),
                "agency": str(r.get("managing_agency") or ""),
                "n_factors": int(r["n_factors"]),
                "split": str(r.get("split") or ""),
                "alert": "SEVERE",
            }
        )

    n_severe = int(pids["severe_delay"].sum())
    return {
        "product": "DRIN",
        "product_full": "Delay Risk Inference Network",
        "tagline": "Budgeted Bayesian VoI inspection for construction delay risk",
        "counts": {
            "projects": int(pids.shape[0]),
            "train": int((pids["split"] == "train").sum()),
            "val": int((pids["split"] == "val").sum()),
            "test": int((pids["split"] == "test").sum()),
            "severe_rate": float(pids["severe_delay"].mean()),
            "severe_projects": n_severe,
            "episode_rows": int(len(ep)),
            "nonempty_reason_rows": int(sum(x["rows"] for x in factor_counts)),
        },
        "factor_counts": factor_counts,
        "severe_watchlist": severe_watchlist,
        "champion": {
            "accuracy": test_m.get("accuracy"),
            "roc_auc": test_m.get("roc_auc"),
            "precision": test_m.get("precision"),
            "recall": test_m.get("recall"),
            "f1": test_m.get("f1"),
            "kappa": test_m.get("kappa"),
            "threshold": lit.get("threshold") or champion.get("threshold"),
            "early_frac": lit.get("early_frac"),
            "model": lit.get("selected_model") or champion.get("selected_model"),
        },
        "literature": lit.get("literature_comparison")
        or champion.get("literature_comparison")
        or [],
        "scoreboard_b2": scoreboard or [],
        "learned_cpts": {
            "p_cause": learned.get("p_cause"),
            "leak": learned.get("leak"),
        },
        "thresholds": thresholds.get("policies", thresholds),
        "beats": {
            "khodabakhshian_85": bool(lit.get("beats_khodabakhshian_85", (test_m.get("accuracy") or 0) > 0.85)),
            "tagharobi_89": bool(lit.get("beats_tagharobi_89", (test_m.get("accuracy") or 0) > 0.89)),
        },
    }


def list_projects(
    q: str = "",
    split: str | None = None,
    severe: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    ep = load_episodes()
    agg: dict[str, tuple] = {
        "severe_delay": ("severe_delay", "max"),
        "split": ("split", "first"),
        "n_updates": ("factor_id", "count"),
        "n_factors": ("factor_id", "nunique"),
    }
    if "phase" in ep.columns:
        agg["phase"] = ("phase", "last")
    if "managing_agency" in ep.columns:
        agg["managing_agency"] = ("managing_agency", "last")
    g = ep.groupby("pid").agg(**agg).reset_index()
    if "phase" not in g.columns:
        g["phase"] = ""
    if "managing_agency" not in g.columns:
        g["managing_agency"] = ""
    if q:
        ql = q.lower()
        g = g[g["pid"].str.lower().str.contains(ql, na=False)]
    if split:
        g = g[g["split"] == split]
    if severe is not None:
        g = g[g["severe_delay"] == int(severe)]
    total = int(len(g))
    page = g.iloc[offset : offset + limit]
    early = load_early_warning()
    rows = []
    for _, r in page.iterrows():
        item = {
            "pid": str(r["pid"]),
            "severe_delay": int(r["severe_delay"]),
            "split": str(r["split"]),
            "n_updates": int(r["n_updates"]),
            "n_factors": int(r["n_factors"]),
            "phase": str(r.get("phase", "")),
            "managing_agency": str(r.get("managing_agency", "")),
        }
        if early is not None:
            er = early[early["pid"] == str(r["pid"])]
            if len(er):
                item["early_pos_var_sum"] = float(er.iloc[0].get("early_pos_var_sum", 0) or 0)
        rows.append(item)
    return {"total": total, "offset": offset, "limit": limit, "items": rows}


def project_detail(pid: str) -> dict[str, Any]:
    ep = load_episodes()
    rows = ep[ep["pid"] == str(pid)]
    if rows.empty:
        raise KeyError(pid)
    factors = factor_ids()
    present = sorted({str(f) for f in rows["factor_id"].unique().tolist() if str(f) in factors})
    timeline = []
    for _, r in rows.sort_values("reporting_period" if "reporting_period" in rows.columns else rows.columns[0]).iterrows():
        timeline.append(
            {
                "reporting_period": float(r["reporting_period"]) if "reporting_period" in rows.columns and pd.notna(r["reporting_period"]) else None,
                "factor_id": str(r["factor_id"]) if pd.notna(r.get("factor_id")) else None,
                "variance_day": float(r["variance_day"]) if "variance_day" in rows.columns and pd.notna(r.get("variance_day")) else None,
                "reason": str(r["reason"]) if "reason" in rows.columns and pd.notna(r.get("reason")) else "",
                "phase": str(r["phase"]) if "phase" in rows.columns and pd.notna(r.get("phase")) else "",
            }
        )
    champ = None
    bundle = load_champion_bundle()
    early = load_early_warning()
    if bundle and early is not None:
        er = early[early["pid"] == str(pid)]
        if len(er):
            feats = bundle["num_cols"] + bundle["cat_cols"]
            X = er[feats]
            prob = float(bundle["model"].predict_proba(X)[:, 1][0])
            thr = float(bundle["threshold"])
            champ = {"p_severe": prob, "threshold": thr, "alert": bool(prob >= thr)}
    return {
        "pid": str(pid),
        "severe_delay": int(rows["severe_delay"].iloc[0]),
        "split": str(rows["split"].iloc[0]),
        "present_factors": present,
        "n_updates": int(len(rows)),
        "timeline": timeline[:80],
        "champion": champ,
        "factor_catalog": get_factors(),
    }


def _make_policy(name: str, order: list[str]) -> Policy:
    if name == "fixed":
        return FixedRankingPolicy(order)
    if name == "static_bn":
        return StaticBNPolicy(order)
    if name == "consensus_voi":
        return ConsensusVoIPolicy(lookahead=1)
    if name == "disagreement_voi":
        return DisagreementVoIPolicy(lookahead=1)
    raise ValueError(f"Unknown policy: {name}")


def _bn_for_policy(name: str, priors) -> DelayRiskBN:
    learned = load_learned_cpts() or {}
    p_cause = float(learned.get("p_cause", 0.55))
    leak = float(learned.get("leak", 0.02))
    if name == "disagreement_voi":
        return DelayRiskBN(factor_priors=priors.disagreement, p_cause=p_cause, leak=leak)
    return DelayRiskBN(factor_priors=priors.consensus, p_cause=p_cause, leak=leak)


def run_inspection(pid: str, policy: str = "disagreement_voi", budget: int = 2) -> dict[str, Any]:
    ep = load_episodes()
    rows = ep[ep["pid"] == str(pid)]
    if rows.empty:
        raise KeyError(pid)
    factors = factor_ids()
    priors = priors_from_episodes(ep, factors)
    order = sorted(factors, key=lambda f: priors.consensus[f], reverse=True)
    thresholds = _read_json(PROC / "calibrated_thresholds.json", {})
    pol_thr = (
        thresholds.get("policies", {})
        .get(policy, {})
        .get("threshold", 0.57)
    )
    pol = _make_policy(policy, order)
    bn = _bn_for_policy(policy, priors)

    # Step-level replay for the UI (mirror episode_replay semantics)
    present_factors = sorted({str(f) for f in rows["factor_id"].unique().tolist() if str(f) in factors})
    work = bn.clone_fresh()
    for f in factors:
        if f not in present_factors:
            work.observe(f, False)
    remaining = set(present_factors)
    state = PolicyState(bn=work, remaining=remaining, budget_left=min(int(budget), len(remaining)))
    steps = []
    prior_p = float(work.query().p_severe)
    steps.append(
        {
            "step": 0,
            "action": "init",
            "factor": None,
            "present": None,
            "p_severe": prior_p,
            "alert": should_alert(prior_p, threshold=float(pol_thr)),
            "budget_left": state.budget_left,
            "remaining": sorted(state.remaining),
        }
    )
    while state.budget_left > 0 and state.remaining:
        choice = pol.select_next(state)
        if choice is None:
            break
        pol.update(state, choice, True)
        p = float(state.bn.query().p_severe)
        steps.append(
            {
                "step": len(steps),
                "action": "inspect",
                "factor": choice,
                "factor_label": FACTOR_LABELS.get(choice, choice),
                "present": True,
                "p_severe": p,
                "alert": should_alert(p, threshold=float(pol_thr)),
                "budget_left": state.budget_left,
                "remaining": sorted(state.remaining),
            }
        )

    # Matched baselines first-pick
    baselines = {}
    for bname in ["fixed", "static_bn", "consensus_voi", "disagreement_voi"]:
        bpol = _make_policy(bname, order)
        bbn = _bn_for_policy(bname, priors).clone_fresh()
        for f in factors:
            if f not in present_factors:
                bbn.observe(f, False)
        bstate = PolicyState(bn=bbn, remaining=set(present_factors), budget_left=min(int(budget), len(present_factors)))
        baselines[bname] = bpol.select_next(bstate)

    final = steps[-1]
    detail = project_detail(pid)
    return {
        "pid": str(pid),
        "policy": policy,
        "budget": int(budget),
        "threshold": float(pol_thr),
        "severe_delay": int(rows["severe_delay"].iloc[0]),
        "present_factors": present_factors,
        "steps": steps,
        "final_p_severe": final["p_severe"],
        "alerted": final["alert"],
        "inspections": [s["factor"] for s in steps if s["action"] == "inspect"],
        "baselines_first_pick": baselines,
        "champion": detail.get("champion"),
        "correct": bool(final["alert"]) == bool(int(rows["severe_delay"].iloc[0])),
    }


def project_options(split: str | None = "test", limit: int = 400) -> dict[str, Any]:
    """Dropdown-friendly project catalog with display names."""
    ep = load_episodes()
    cols = ["pid", "severe_delay", "split"]
    for c in ["project_name", "managing_agency", "phase"]:
        if c in ep.columns:
            cols.append(c)
    g = ep.groupby("pid").agg(
        **{
            "severe_delay": ("severe_delay", "max"),
            "split": ("split", "first"),
            **({"project_name": ("project_name", "last")} if "project_name" in ep.columns else {}),
            **({"managing_agency": ("managing_agency", "last")} if "managing_agency" in ep.columns else {}),
            **({"phase": ("phase", "last")} if "phase" in ep.columns else {}),
            "n_factors": ("factor_id", "nunique"),
        }
    ).reset_index()
    if split:
        g = g[g["split"] == split]
    g = g.head(limit)
    items = []
    for _, r in g.iterrows():
        name = str(r.get("project_name") or "").strip() or f"Project {r['pid']}"
        agency = str(r.get("managing_agency") or "").strip()
        label = f"{name}  ·  PID {r['pid']}" + (f"  ·  {agency}" if agency else "")
        items.append(
            {
                "pid": str(r["pid"]),
                "label": label,
                "name": name,
                "agency": agency,
                "phase": str(r.get("phase") or ""),
                "severe_delay": int(r["severe_delay"]),
                "split": str(r["split"]),
                "n_factors": int(r["n_factors"]),
            }
        )
    items.sort(key=lambda x: x["name"].lower())
    return {"items": items, "total": len(items)}


def _inspect_from_present_factors(
    present_factors: list[str],
    policy: str,
    budget: int,
    *,
    pid: str | None = None,
    severe_delay: int | None = None,
    source: str = "registry",
    mapped_reasons: list[dict] | None = None,
    project_name: str | None = None,
) -> dict[str, Any]:
    ep = load_episodes()
    factors = factor_ids()
    priors = priors_from_episodes(ep, factors)
    order = sorted(factors, key=lambda f: priors.consensus[f], reverse=True)
    thresholds = _read_json(PROC / "calibrated_thresholds.json", {})
    pol_thr = float(thresholds.get("policies", {}).get(policy, {}).get("threshold", 0.57))
    pol = _make_policy(policy, order)
    bn = _bn_for_policy(policy, priors)

    present = sorted([f for f in present_factors if f in factors])
    work = bn.clone_fresh()
    for f in factors:
        if f not in present:
            work.observe(f, False)

    remaining = set(present)
    state = PolicyState(bn=work, remaining=remaining, budget_left=min(int(budget), max(len(remaining), 0)))
    steps = []
    prior_p = float(work.query().p_severe)
    steps.append(
        {
            "step": 0,
            "action": "init",
            "factor": None,
            "present": None,
            "p_severe": prior_p,
            "alert": should_alert(prior_p, threshold=pol_thr),
            "budget_left": state.budget_left,
            "remaining": sorted(state.remaining),
            "delta": 0.0,
        }
    )
    while state.budget_left > 0 and state.remaining:
        choice = pol.select_next(state)
        if choice is None:
            break
        before = float(state.bn.query().p_severe)
        pol.update(state, choice, True)
        p = float(state.bn.query().p_severe)
        steps.append(
            {
                "step": len(steps),
                "action": "inspect",
                "factor": choice,
                "factor_label": FACTOR_LABELS.get(choice, choice),
                "present": True,
                "p_severe": p,
                "alert": should_alert(p, threshold=pol_thr),
                "budget_left": state.budget_left,
                "remaining": sorted(state.remaining),
                "delta": p - before,
            }
        )

    finals = steps[-1]
    # Factor influence: rise in P when that factor alone is revealed (myopic)
    influence = []
    base = DelayRiskBN(
        factor_priors=(_bn_for_policy(policy, priors).factor_priors),
        p_cause=_bn_for_policy(policy, priors).p_cause,
        leak=_bn_for_policy(policy, priors).leak,
    )
    for f in factors:
        if f not in present:
            base.observe(f, False)
    p0 = float(base.query().p_severe)
    for f in present:
        twin = base.clone_fresh()
        for g in factors:
            if g not in present:
                twin.observe(g, False)
            elif g == f:
                twin.observe(g, True)
        influence.append(
            {
                "factor": f,
                "label": FACTOR_LABELS.get(f, f),
                "lift": float(twin.query().p_severe) - p0,
                "inspected": any(s.get("factor") == f for s in steps),
            }
        )
    influence.sort(key=lambda x: -abs(x["lift"]))

    champion = None
    if pid:
        try:
            champion = project_detail(pid).get("champion")
        except Exception:
            champion = None

    overview = get_overview()
    alerted = bool(finals["alert"])
    truth = None if severe_delay is None else int(severe_delay)
    return {
        "ok": True,
        "source": source,
        "pid": pid,
        "project_name": project_name,
        "policy": policy,
        "budget": int(budget),
        "threshold": pol_thr,
        "severe_delay": truth,
        "present_factors": present,
        "mapped_reasons": mapped_reasons or [],
        "steps": steps,
        "probability_series": [{"step": s["step"], "p_severe": s["p_severe"], "label": s.get("factor_label") or s["action"]} for s in steps],
        "factor_influence": influence,
        "final_p_severe": float(finals["p_severe"]),
        "alerted": alerted,
        "inspections": [s["factor"] for s in steps if s["action"] == "inspect"],
        "champion": champion,
        "correct": (None if truth is None else alerted == bool(truth)),
        "computation": {
            "stages": [
                {"id": "ingest", "label": "Ingest evidence", "detail": f"{len(present)} active factors"},
                {"id": "map", "label": "Map reasons → ontology", "detail": f"{len(mapped_reasons or [])} reason spans"},
                {"id": "bn", "label": "Bayesian Noisy-OR update", "detail": f"p_cause={overview['learned_cpts'].get('p_cause')}"},
                {"id": "voi", "label": f"Budgeted VoI ({policy})", "detail": f"B={budget}"},
                {"id": "decide", "label": "Alert decision", "detail": f"τ={pol_thr:.3f}"},
            ]
        },
        "system_benchmarks": {
            "champion_accuracy": overview["champion"].get("accuracy"),
            "champion_auc": overview["champion"].get("roc_auc"),
            "beats_tagharobi_89": overview["beats"]["tagharobi_89"],
            "beats_khodabakhshian_85": overview["beats"]["khodabakhshian_85"],
        },
        "factor_catalog": get_factors(),
    }


def run_analysis(
    *,
    pid: str | None = None,
    budget: int = 2,
    policy: str = "disagreement_voi",
    pdf_bytes: bytes | None = None,
    manual_text: str | None = None,
) -> dict[str, Any]:
    """Unified analysis entry: registry project and/or PDF/manual text."""
    from app.pdf_ingest import extract_pdf_text, map_reasons_to_factors, reason_candidates

    mapped_reasons: list[dict] = []
    present_from_doc: list[str] = []
    pdf_meta = None

    raw_text = (manual_text or "").strip()
    if pdf_bytes:
        text = extract_pdf_text(pdf_bytes)
        pdf_meta = {"chars": len(text), "preview": text[:600]}
        raw_text = (raw_text + "\n" + text).strip() if raw_text else text

    if raw_text:
        reasons = reason_candidates(raw_text)
        mapped = map_reasons_to_factors(reasons)
        mapped_reasons = mapped["reasons"]
        present_from_doc = mapped["present_factors"]

    if pid:
        base = run_inspection(pid, policy=policy, budget=budget)
        # Merge PDF-derived factors into present set for richer board (inspection still from episode)
        detail = project_detail(pid)
        present = sorted(set(base["present_factors"]) | set(present_from_doc))
        enriched = _inspect_from_present_factors(
            present,
            policy,
            budget,
            pid=pid,
            severe_delay=base["severe_delay"],
            source="registry+document" if present_from_doc else "registry",
            mapped_reasons=mapped_reasons,
            project_name=None,
        )
        # Prefer episode-faithful inspection path from run_inspection for registry truth
        # but keep influence + mapped reasons from enriched
        ep = load_episodes()
        rows = ep[ep["pid"] == str(pid)]
        name = ""
        if "project_name" in rows.columns and len(rows):
            name = str(rows["project_name"].iloc[-1] or "")
        out = {**base, **{
            "ok": True,
            "source": enriched["source"],
            "project_name": name or f"Project {pid}",
            "mapped_reasons": mapped_reasons,
            "pdf": pdf_meta,
            "probability_series": [
                {"step": s["step"], "p_severe": s["p_severe"], "label": s.get("factor_label") or s["action"]}
                for s in base["steps"]
            ],
            "factor_influence": enriched["factor_influence"],
            "computation": enriched["computation"],
            "system_benchmarks": enriched["system_benchmarks"],
            "factor_catalog": get_factors(),
            "manual_or_pdf_factors": present_from_doc,
        }}
        # rebuild steps with delta
        series = []
        prev = None
        for s in base["steps"]:
            d = 0.0 if prev is None else float(s["p_severe"]) - prev
            prev = float(s["p_severe"])
            series.append({**s, "delta": d})
        out["steps"] = series
        out["probability_series"] = [
            {"step": s["step"], "p_severe": s["p_severe"], "label": s.get("factor_label") or s["action"], "delta": s.get("delta", 0)}
            for s in series
        ]
        return out

    if not present_from_doc:
        raise ValueError("Provide a project from the registry and/or a PDF / delay-notes text with extractable reasons.")

    return {
        **_inspect_from_present_factors(
            present_from_doc,
            policy,
            budget,
            pid=None,
            severe_delay=None,
            source="document",
            mapped_reasons=mapped_reasons,
            project_name="Uploaded document case",
        ),
        "pdf": pdf_meta,
        "manual_or_pdf_factors": present_from_doc,
    }


def get_scenarios() -> dict[str, Any]:
    path = DEMO / "case_scenarios.json"
    if not path.exists():
        return {"scenarios": []}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for key, val in raw.items():
        out.append({"id": key, **val})
    return {"scenarios": out}


def run_scenario(scenario_id: str, inspections: list[str], budget: int = 2) -> dict[str, Any]:
    scenarios = {s["id"]: s for s in get_scenarios()["scenarios"]}
    if scenario_id not in scenarios:
        raise KeyError(scenario_id)
    sc = scenarios[scenario_id]
    factors = factor_ids()
    allowed = set(sc.get("inspectable_factors") or factors)
    chosen = [f for f in inspections if f in allowed][:budget]
    # Synthetic BN with elevated priors on inspectable factors
    priors = {f: 0.15 for f in factors}
    for f in allowed:
        priors[f] = 0.35
    bn = DelayRiskBN(factor_priors=priors, p_cause=0.55, leak=0.02)
    for f in factors:
        if f not in allowed:
            bn.observe(f, False)
    steps = [{"step": 0, "factor": None, "p_severe": float(bn.query().p_severe), "present": None}]
    for i, f in enumerate(chosen, start=1):
        # In narrative demos, inspecting a listed factor reveals it present
        bn.observe(f, True)
        steps.append(
            {
                "step": i,
                "factor": f,
                "factor_label": FACTOR_LABELS.get(f, f),
                "present": True,
                "p_severe": float(bn.query().p_severe),
            }
        )
    final_p = steps[-1]["p_severe"]
    return {
        "scenario": sc,
        "inspections": chosen,
        "steps": steps,
        "final_p_severe": final_p,
        "alerted": final_p >= 0.55,
        "truth_severe": bool(sc.get("severe_delay")),
    }
