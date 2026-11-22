"""DRIN FastAPI application — Delay Risk Inference Network."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import services
from app.pdf_ingest import extract_pdf_text, map_reasons_to_factors, reason_candidates

app = FastAPI(
    title="DRIN",
    version="1.0.0",
    description="Delay Risk Inference Network — budgeted Bayesian VoI inspection",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InspectRequest(BaseModel):
    pid: str
    policy: str = "disagreement_voi"
    budget: int = Field(default=2, ge=1, le=4)


class ScenarioInspectRequest(BaseModel):
    inspections: list[str] = Field(default_factory=list)
    budget: int = Field(default=2, ge=1, le=4)


@app.get("/api/health")
def health():
    return {"ok": True, "product": "DRIN", "product_full": "Delay Risk Inference Network"}


@app.get("/api/overview")
def overview():
    try:
        return services.get_overview()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e


@app.get("/api/factors")
def factors():
    return {"factors": services.get_factors()}


@app.get("/api/projects/options")
def project_options(split: str | None = "test", limit: int = Query(default=400, ge=1, le=2000)):
    try:
        return services.project_options(split=split, limit=limit)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e


@app.get("/api/projects")
def projects(
    q: str = "",
    split: str | None = None,
    severe: int | None = Query(default=None),
    limit: int = Query(default=40, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    try:
        return services.list_projects(q=q, split=split, severe=severe, limit=limit, offset=offset)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e


@app.get("/api/projects/{pid}")
def project(pid: str):
    try:
        return services.project_detail(pid)
    except KeyError as e:
        raise HTTPException(404, f"Unknown PID {pid}") from e
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e


@app.post("/api/inspect/run")
def inspect_run(body: InspectRequest):
    try:
        return services.run_inspection(body.pid, policy=body.policy, budget=body.budget)
    except KeyError as e:
        raise HTTPException(404, f"Unknown PID {body.pid}") from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e


@app.post("/api/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Upload a .pdf file")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty PDF")
    try:
        text = extract_pdf_text(data)
        reasons = reason_candidates(text)
        mapped = map_reasons_to_factors(reasons)
        return {
            "filename": file.filename,
            "chars": len(text),
            "preview": text[:800],
            **mapped,
        }
    except Exception as e:
        raise HTTPException(400, f"Could not parse PDF: {e}") from e


@app.post("/api/analyze")
async def analyze(
    pid: str | None = Form(default=None),
    budget: int = Form(default=2),
    policy: str = Form(default="disagreement_voi"),
    notes: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
):
    if budget < 1 or budget > 4:
        raise HTTPException(400, "budget must be 1–4")
    pdf_bytes = None
    if file is not None and file.filename:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Only PDF uploads are supported")
        pdf_bytes = await file.read()
    try:
        return services.run_analysis(
            pid=(pid or None) or None,
            budget=budget,
            policy=policy,
            pdf_bytes=pdf_bytes,
            manual_text=notes,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e


@app.get("/api/scenarios")
def scenarios():
    return services.get_scenarios()


@app.post("/api/scenarios/{scenario_id}/run")
def scenario_run(scenario_id: str, body: ScenarioInspectRequest):
    try:
        return services.run_scenario(scenario_id, body.inspections, budget=body.budget)
    except KeyError as e:
        raise HTTPException(404, f"Unknown scenario {scenario_id}") from e


WEB_DIST = ROOT / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str = ""):
        index = WEB_DIST / "index.html"
        candidate = WEB_DIST / full_path
        if full_path and candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)
