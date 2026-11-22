import { useEffect, useState } from 'react'
import { api, pct } from '../api'

type Scenario = {
  id: string
  name: string
  severe_delay: boolean
  actual_overrun_pct: number
  inspectable_factors: string[]
  notes?: string
}

export function Scenarios() {
  const [scenarios, setScenarios] = useState<Scenario[]>([])
  const [active, setActive] = useState<string>('')
  const [picked, setPicked] = useState<string[]>([])
  const [budget, setBudget] = useState(2)
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [labels, setLabels] = useState<Record<string, string>>({})

  useEffect(() => {
    api.scenarios().then((r) => {
      const list = r.scenarios as Scenario[]
      setScenarios(list)
      if (list[0]) setActive(list[0].id)
    })
    api.factors().then((r) => {
      const m: Record<string, string> = {}
      r.factors.forEach((f) => {
        m[f.id] = f.label
      })
      setLabels(m)
    })
  }, [])

  const sc = scenarios.find((s) => s.id === active)

  function toggle(f: string) {
    setPicked((prev) => {
      if (prev.includes(f)) return prev.filter((x) => x !== f)
      if (prev.length >= budget) return [...prev.slice(1), f]
      return [...prev, f]
    })
  }

  async function run() {
    if (!active) return
    const res = await api.runScenario(active, picked, budget)
    setResult(res)
  }

  return (
    <>
      <header className="page-head">
        <div>
          <h1>Scenarios</h1>
          <p>
            Chen narrative walkthroughs for viva demos. Pick up to B factors, then update the
            posterior.
          </p>
        </div>
      </header>

      <div className="scenario-pick" style={{ marginBottom: '1rem' }}>
        {scenarios.map((s) => (
          <button
            key={s.id}
            className={active === s.id ? 'selected' : ''}
            onClick={() => {
              setActive(s.id)
              setPicked([])
              setResult(null)
            }}
          >
            <strong>{s.name}</strong>
            <div style={{ marginTop: '0.35rem', color: 'var(--slate)', fontSize: '0.85rem' }}>
              overrun {s.actual_overrun_pct}% ·{' '}
              {s.severe_delay ? 'severe class' : 'non-severe class'}
            </div>
          </button>
        ))}
      </div>

      {sc && (
        <div className="grid grid-2">
          <section className="panel">
            <h2>Inspection tokens (B = {budget})</h2>
            <div className="field" style={{ marginBottom: '0.85rem', maxWidth: 140 }}>
              <label>Budget</label>
              <select value={budget} onChange={(e) => setBudget(Number(e.target.value))}>
                {[1, 2, 3, 4].map((b) => (
                  <option key={b} value={b}>
                    {b}
                  </option>
                ))}
              </select>
            </div>
            <div className="factor-board">
              {sc.inspectable_factors.map((f) => (
                <button
                  key={f}
                  className={`factor-chip ${picked.includes(f) ? 'inspected' : 'present'}`}
                  onClick={() => toggle(f)}
                  type="button"
                >
                  <div className="fid">{f}</div>
                  <div className="flabel">{labels[f] || f}</div>
                </button>
              ))}
            </div>
            <button className="btn btn-primary" style={{ marginTop: '1rem' }} onClick={run}>
              Reveal posterior
            </button>
            <p style={{ color: 'var(--slate)', fontSize: '0.85rem' }}>{sc.notes}</p>
          </section>

          <section className="panel">
            <h2>Outcome</h2>
            {!result && <div className="empty">Select factors and reveal.</div>}
            {result && (
              <>
                <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.85rem' }}>
                  <span className="badge badge-muted">
                    P(severe) = {pct(Number(result.final_p_severe))}
                  </span>
                  {result.alerted ? (
                    <span className="badge badge-danger">alert</span>
                  ) : (
                    <span className="badge badge-ok">hold</span>
                  )}
                  {result.truth_severe ? (
                    <span className="badge badge-warn">truth severe</span>
                  ) : (
                    <span className="badge badge-muted">truth normal</span>
                  )}
                </div>
                <div className="timeline">
                  {(result.steps as Array<Record<string, unknown>>).map((s) => (
                    <div className="step-row" key={String(s.step)}>
                      <div className="n">{String(s.step)}</div>
                      <div>
                        <strong>
                          {s.factor
                            ? `Inspect ${String(s.factor_label || s.factor)}`
                            : 'Prior'}
                        </strong>
                      </div>
                      <div style={{ fontFamily: 'var(--font-brand)', fontWeight: 700 }}>
                        {pct(Number(s.p_severe))}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </section>
        </div>
      )}
    </>
  )
}
