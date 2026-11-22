import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api, num, pct } from '../api'
import type { AnalysisResult, ProjectOption } from '../api'

type Phase = 'form' | 'running' | 'done'

const POLICIES = [
  { id: 'disagreement_voi', label: 'Disagreement-aware VoI' },
  { id: 'consensus_voi', label: 'Consensus VoI' },
  { id: 'static_bn', label: 'Static checklist (BN)' },
  { id: 'fixed', label: 'Fixed ranking' },
]

export function Lab() {
  const [params] = useSearchParams()
  const [options, setOptions] = useState<ProjectOption[]>([])
  const [pid, setPid] = useState(params.get('pid') || '')
  const [budget, setBudget] = useState(2)
  const [policy, setPolicy] = useState('disagreement_voi')
  const [notes, setNotes] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [phase, setPhase] = useState<Phase>('form')
  const [stageIdx, setStageIdx] = useState(0)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    const q = params.get('pid')
    api.options('test').then((r) => {
      setOptions(r.items)
      if (q) setPid(q)
      else if (!pid && r.items[0]) setPid(r.items[0].pid)
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params])

  const selected = useMemo(() => options.find((o) => o.pid === pid), [options, pid])

  async function run() {
    setErr(null)
    if (!pid && !file && !notes.trim()) {
      setErr('Select a project or provide a PDF / delay notes.')
      return
    }
    setPhase('running')
    setStageIdx(0)
    setResult(null)

    const form = new FormData()
    if (pid) form.append('pid', pid)
    form.append('budget', String(budget))
    form.append('policy', policy)
    if (notes.trim()) form.append('notes', notes.trim())
    if (file) form.append('file', file)

    const tick = window.setInterval(() => setStageIdx((i) => Math.min(i + 1, 4)), 380)
    try {
      const res = await api.analyze(form)
      setStageIdx(4)
      await new Promise((r) => setTimeout(r, 280))
      setResult(res)
      setPhase('done')
    } catch (e: unknown) {
      setErr(String((e as Error).message || e))
      setPhase('form')
    } finally {
      window.clearInterval(tick)
    }
  }

  const stages = [
    { t: 'Load project evidence', d: 'Schedule history & delay notes' },
    { t: 'Classify delay drivers', d: 'Map text → 8 risk categories' },
    { t: 'Update risk model', d: 'Bayesian Noisy-OR posterior' },
    { t: 'Allocate inspections', d: `Value of Information · budget ${budget}` },
    { t: 'Issue recommendation', d: 'Alert vs hold against threshold' },
  ]

  return (
    <>
      <header className="ph">
        <div>
          <h1>Project review</h1>
          <p>
            Run a budgeted inspection on a capital project. The system chooses which delay drivers
            to verify next and updates severe-schedule risk.
          </p>
        </div>
        <Link className="btn btn-ghost" to="/">
          Back to portfolio
        </Link>
      </header>

      {phase !== 'done' && (
        <section className="panel">
          <h2>Review setup</h2>
          <div className="form-grid">
            <div className="field">
              <label>Project</label>
              <select value={pid} onChange={(e) => setPid(e.target.value)}>
                <option value="">Document only</option>
                {options.map((o) => (
                  <option key={o.pid} value={o.pid}>
                    {o.severe_delay ? '● ' : '○ '}
                    {o.label}
                  </option>
                ))}
              </select>
              {selected && (
                <div className="meta" style={{ marginTop: '0.4rem' }}>
                  {selected.severe_delay ? (
                    <span className="tag tag-risk">At risk</span>
                  ) : (
                    <span className="tag tag-ok">On track</span>
                  )}{' '}
                  · {selected.phase || 'Phase n/a'} · {selected.n_factors} drivers on record
                </div>
              )}
            </div>
            <div className="field">
              <label>Inspection budget</label>
              <select value={budget} onChange={(e) => setBudget(Number(e.target.value))}>
                {[1, 2, 3, 4].map((b) => (
                  <option key={b} value={b}>
                    {b} check{b > 1 ? 's' : ''}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label>Decision method</label>
              <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
                {POLICIES.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="layout-2" style={{ marginTop: '1rem' }}>
            <div
              className={`drop ${file ? 'has' : ''}`}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault()
                const f = e.dataTransfer.files?.[0]
                if (f?.name.toLowerCase().endsWith('.pdf')) setFile(f)
              }}
            >
              <strong>{file ? file.name : 'Attach schedule / delay PDF (optional)'}</strong>
              <span>Field reports and change narratives are mapped to risk categories.</span>
              <div style={{ marginTop: '0.65rem' }}>
                <input
                  type="file"
                  accept="application/pdf,.pdf"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                />
              </div>
            </div>
            <div className="field">
              <label>Delay notes (optional)</label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Utility work delayed foundation. Scope redesign on level 3…"
              />
            </div>
          </div>

          <div className="form-actions">
            <button className="btn btn-accent" disabled={phase === 'running'} onClick={run}>
              {phase === 'running' ? 'Running…' : 'Run inspection'}
            </button>
            <button
              className="btn btn-ghost"
              type="button"
              onClick={() => {
                setFile(null)
                setNotes('')
                setErr(null)
              }}
            >
              Clear attachments
            </button>
          </div>
          {err && <div className="error" style={{ marginTop: '0.85rem' }}>{err}</div>}
        </section>
      )}

      {phase === 'running' && (
        <section className="panel">
          <h2>Running analysis</h2>
          <div className="progress">
            <i style={{ width: `${((stageIdx + 1) / stages.length) * 100}%` }} />
          </div>
          <div className="steps">
            {stages.map((s, i) => (
              <div key={s.t} className={`step ${i < stageIdx ? 'done' : ''} ${i === stageIdx ? 'on' : ''}`}>
                <div className="n">{i + 1}</div>
                <div>
                  <div className="t">{s.t}</div>
                  <div className="d">{s.d}</div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {phase === 'done' && result && (
        <>
          <div className={`result-head ${result.alerted ? 'alert-on' : ''}`}>
            <div>
              <h2>{result.project_name || 'Document review'}</h2>
              <p>
                {result.policy.replaceAll('_', ' ')} · budget {result.budget} · source {result.source}
              </p>
              <div style={{ marginTop: '0.65rem', display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                {result.alerted ? (
                  <span className="tag tag-risk">Recommend escalate</span>
                ) : (
                  <span className="tag tag-ok">Hold / monitor</span>
                )}
                {result.severe_delay == null ? null : result.severe_delay ? (
                  <span className="tag tag-risk">Recorded as at-risk</span>
                ) : (
                  <span className="tag tag-blue">Recorded on-track</span>
                )}
              </div>
            </div>
            <div className={`big-metric ${result.alerted ? 'hot' : ''}`}>
              <div className="lbl">P(severe delay)</div>
              <div className="num">{pct(result.final_p_severe, 0)}</div>
            </div>
          </div>

          <div className="form-actions" style={{ marginTop: 0, marginBottom: '1rem' }}>
            <button className="btn btn-primary" onClick={() => setPhase('form')}>
              New review
            </button>
            <Link className="btn btn-ghost" to="/">
              Portfolio
            </Link>
          </div>

          <div className="metrics">
            <div className="metric risk">
              <div className="lbl">Checks used</div>
              <div className="num">
                {result.inspections.length}/{result.budget}
              </div>
              <div className="sub">{result.inspections.join(', ') || 'none'}</div>
            </div>
            <div className="metric primary">
              <div className="lbl">Alert threshold</div>
              <div className="num" style={{ fontSize: '1.5rem' }}>
                {num(result.threshold, 2)}
              </div>
              <div className="sub">Validation-calibrated</div>
            </div>
            <div className="metric">
              <div className="lbl">Early-warning model</div>
              <div className="num" style={{ fontSize: '1.5rem' }}>
                {result.champion ? pct(result.champion.p_severe, 0) : '—'}
              </div>
              <div className="sub">
                {result.champion?.alert ? 'Flags project' : result.champion ? 'Clear' : 'n/a'}
              </div>
            </div>
            <div className="metric">
              <div className="lbl">Decision quality</div>
              <div className="num" style={{ fontSize: '1.35rem' }}>
                {result.correct == null ? 'n/a' : result.correct ? 'Match' : 'Miss'}
              </div>
              <div className="sub">Vs recorded severe label</div>
            </div>
          </div>

          <div className="layout-2">
            <section className="panel">
              <h2>Risk trajectory</h2>
              <div className="chart-lg">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={result.probability_series}>
                    <CartesianGrid stroke="#e6ebf0" strokeDasharray="3 3" />
                    <XAxis dataKey="step" tick={{ fontSize: 12 }} />
                    <YAxis
                      domain={[0, 1]}
                      tickFormatter={(v) => `${Math.round(Number(v) * 100)}%`}
                      tick={{ fontSize: 12 }}
                    />
                    <Tooltip formatter={(v) => pct(Number(v))} />
                    <Area
                      type="monotone"
                      dataKey="p_severe"
                      stroke="#0b5dd0"
                      fill="#bfd2e4"
                      strokeWidth={2.5}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="steps" style={{ marginTop: '0.85rem' }}>
                {result.steps.map((s) => (
                  <div key={s.step} className="step done">
                    <div className="n">{s.step}</div>
                    <div>
                      <div className="t">
                        {s.action === 'init'
                          ? 'Baseline after known-absent clamps'
                          : `Inspected ${s.factor_label || s.factor}`}
                      </div>
                      <div className="d">
                        P = {pct(s.p_severe)}
                        {typeof s.delta === 'number' && s.step > 0
                          ? ` · Δ ${s.delta >= 0 ? '+' : ''}${num(s.delta, 3)}`
                          : ''}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <div>
              <section className="panel">
                <h2>Driver impact</h2>
                <div className="chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={result.factor_influence.slice(0, 8)}
                      layout="vertical"
                      margin={{ left: 4, right: 8 }}
                    >
                      <CartesianGrid stroke="#e6ebf0" strokeDasharray="3 3" horizontal={false} />
                      <XAxis type="number" tickFormatter={(v) => pct(Number(v), 0)} tick={{ fontSize: 11 }} />
                      <YAxis type="category" dataKey="label" width={110} tick={{ fontSize: 11 }} />
                      <Tooltip formatter={(v) => num(Number(v), 3)} />
                      <Bar dataKey="lift" barSize={12}>
                        {result.factor_influence.slice(0, 8).map((f) => (
                          <Cell key={f.factor} fill={f.inspected ? '#ff4026' : '#0b5dd0'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <p className="panel-note">Orange = inspected under budget</p>
              </section>

              <section className="panel">
                <h2>Document mapping</h2>
                {result.mapped_reasons?.length ? (
                  <div className="reason-box">
                    {result.mapped_reasons.slice(0, 12).map((r, i) => (
                      <div className="row" key={`${i}-${r.factor_id}`}>
                        <code>{r.factor_id}</code>
                        <div>{r.reason}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty">No PDF/notes attached — used project record only.</div>
                )}
              </section>
            </div>
          </div>
        </>
      )}
    </>
  )
}
