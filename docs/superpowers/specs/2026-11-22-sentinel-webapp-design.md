# SENTINEL Web Application Design

**Product:** SENTINEL — Sequential Inspection for Delay Risk  
**Decision:** Replace Streamlit with FastAPI + React (Vite) official software UI.  
**Surfaces:** Overview dashboard + deep inspection workspace (+ scenarios, literature).

## Architecture

- `app/api.py` — FastAPI: projects, inspect, overview, scenarios, literature
- `web/` — React + TypeScript SPA (Vite)
- Core math stays in `src/` (BN, VoI, ML); API is a thin orchestration layer
- Streamlit `demo/app.py` removed

## Pages

1. **Command** — KPI strip, literature beat, policy scoreboard, system status  
2. **Projects** — searchable PID queue with severe flag / split  
3. **Inspect** — policy + budget console, step animation, risk gauge, factor board  
4. **Scenarios** — Chen theater / airport interactive walkthrough  
5. **Results** — champion metrics + literature table  

## Approval

User directed: fully functional, whatever it takes for 10/10; chose dashboard + inspection (both).
