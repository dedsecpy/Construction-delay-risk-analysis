export type Overview = {
  product: string
  product_full?: string
  tagline: string
  counts: {
    projects: number
    train: number
    val: number
    test: number
    severe_rate: number
    severe_projects?: number
    episode_rows: number
    nonempty_reason_rows?: number
  }
  factor_counts?: Array<{ factor: string; label: string; rows: number; projects: number }>
  severe_watchlist?: Array<{
    pid: string
    name: string
    agency: string
    n_factors: number
    split: string
    alert: string
  }>
  champion: Record<string, number | string | null | undefined>
  literature: Array<{
    literature_metric: string
    literature_value: number
    our_test_accuracy: number
    our_test_auc?: number
    beat: boolean
    margin_accuracy: number
  }>
  scoreboard_b2: Array<Record<string, string | number>>
  learned_cpts: { p_cause?: number; leak?: number }
  beats: { khodabakhshian_85: boolean; tagharobi_89: boolean }
}

export type ProjectOption = {
  pid: string
  label: string
  name: string
  agency: string
  phase: string
  severe_delay: number
  split: string
  n_factors: number
}

export type AnalysisResult = {
  ok: boolean
  source: string
  pid: string | null
  project_name?: string | null
  policy: string
  budget: number
  threshold: number
  severe_delay: number | null
  present_factors: string[]
  mapped_reasons: Array<{ reason: string; factor_id: string }>
  steps: Array<{
    step: number
    action: string
    factor: string | null
    factor_label?: string
    p_severe: number
    alert: boolean
    delta?: number
  }>
  probability_series: Array<{ step: number; p_severe: number; label: string; delta?: number }>
  factor_influence: Array<{ factor: string; label: string; lift: number; inspected: boolean }>
  final_p_severe: number
  alerted: boolean
  inspections: string[]
  champion: { p_severe: number; threshold: number; alert: boolean } | null
  correct: boolean | null
  computation: { stages: Array<{ id: string; label: string; detail: string }> }
  system_benchmarks: {
    champion_accuracy?: number
    champion_auc?: number
    beats_tagharobi_89?: boolean
    beats_khodabakhshian_85?: boolean
  }
  pdf?: { chars: number; preview: string } | null
  factor_catalog: Array<{ id: string; label: string }>
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init)
  if (!res.ok) {
    let msg = res.statusText
    try {
      const j = await res.json()
      msg = j.detail || JSON.stringify(j)
    } catch {
      msg = await res.text()
    }
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
  }
  return res.json() as Promise<T>
}

export const api = {
  overview: () => req<Overview>('/api/overview'),
  options: (split = 'test') =>
    req<{ items: ProjectOption[]; total: number }>(`/api/projects/options?split=${split}&limit=500`),
  analyze: (form: FormData) =>
    req<AnalysisResult>('/api/analyze', { method: 'POST', body: form }),
}

export function pct(n?: number | null, digits = 1) {
  if (n == null || Number.isNaN(n)) return '—'
  return `${(n * 100).toFixed(digits)}%`
}

export function num(n?: number | null, digits = 3) {
  if (n == null || Number.isNaN(n)) return '—'
  return Number(n).toFixed(digits)
}
