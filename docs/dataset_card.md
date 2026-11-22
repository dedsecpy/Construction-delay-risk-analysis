# Dataset card

## Primary — NYC Open Data

| Dataset | ID | Scale | Use |
|---|---|---|---|
| Schedule History & Variance | `95tx-snak` | ~22k rows / ~3.6k PIDs | Sequential evidence + delay reasons |
| Capital Projects Schedules & Budgets | `n7gv-k5yt` | ~3.3k rows / ~363 PIDs | Labels / covariates |

Downloads:
- https://data.cityofnewyork.us/api/v3/views/95tx-snak/export.csv?accessType=DOWNLOAD
- https://data.cityofnewyork.us/api/v3/views/n7gv-k5yt/export.csv?accessType=DOWNLOAD

## Secondary — UK NISTA GMPP 2025–26
~189 major projects with DCA Red/Amber/Green for transfer validation.
https://www.gov.uk/government/publications/nista-major-projects-annual-report-2025-26

## Severe-delay label rule
1. Prefer relative slip ≥ 25% of planned duration when design/start and forecast/actual completion exist.
2. Else mark severe if cumulative positive `variance_day` is in the top quartile among projects with ≥2 history rows.
3. Document which rule applied per PID in processed metadata.

## Factors (8)
Mapped from NYC delay reasons ↔ Chen ontology:
`budget`, `scope_design`, `grants_approvals`, `procurement`, `construction_conditions`, `external_dependency`, `workforce_resource`, `other_admin`