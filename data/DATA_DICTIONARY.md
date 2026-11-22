# Data dictionary

## factors.csv
- `factor_id` — stable node id used in BN/policies
- `name`, `description` — human labels

## edges.csv
- `parent` → `child` directed BN edges

## reason_map.csv
- `pattern` — regex (case-insensitive) matched against NYC delay reason text
- `factor_id` — mapped factor; empty pattern is fallback `other_admin`

## episodes.csv (generated)
- `pid`, `step`, `reporting_period`, `factor_id`, `observed` (0/1)
- `variance_day`, `cumulative_variance`, `severe_delay`, `split`