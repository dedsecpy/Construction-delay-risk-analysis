import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api, num, pct } from '../api'
import type { InspectResult } from '../api'

const POLICIES = [
  { id: 'disagreement_voi', label: 'Disagreement VoI' },
  { id: 'consensus_voi', label: 'Consensus VoI' },
  { id: 'static_bn', label: 'Static BN' },
  { id: 'fixed', label: 'Fixed ranking' },
]

export function Inspect() {
  const [params, setParams] = useSearchParams()
  const [pid, setPid] = useState(params.get('pid') || '')
  const [policy, setPolicy] = useState('disagreement_voi')
  const [budget, setBudget] = useState(2)
  const [pids, setPids] = useState<string[]>([])
  const [result, setResult] = useState<InspectResult | null>(null)
  const [cursor, setCursor] = useState(0)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [factors, setFactors] = useState<Array<{ id: string; label: string }>>([])

  useEffect(() => {
    api.factors().then((r) => setFactors(r.factors))
    const sp = new URLSearchParams({ split: 'test', limit: '120' })
    api.projects(sp).then((r) => setPids(r.items.map((x) => x.pid)))
  }, [])

  useEffect(() => {
    const p = params.get('pid')
    if (p) setPid(p)
  }, [params])

  async function run() {
    if (!pid) return
    setBusy(true)
    setErr(null)
    try {
      const res = await api.inspect({ pid, policy, budget })
      setResult(res)
      setCursor(0)
      setParams({ pid })
      // animate steps
      for (let i = 0; i < res.steps.length; i++) {
        setCursor(i)
        await new Promise((r) => setTimeout(r, 380))
      }
    } catch (e: unknown) {
      setErr(String((e as Error).message || e))
    } finally {
      setBusy(false)
    }
  }

  const step = result?.steps[cursor]
  const p = step?.p_severe ?? 0
  const inspected = useMemo(() => {
    if (!result) return new Set<string>()
    return new Set(
      result.steps.slice(0, cursor + 1).map((s) => s.factor).filter(Boolean) as string[],
    )
  }, [result, cursor])

  return (
    <>
      <header className="page-head">
        <div>
          <h1>Inspect</h1>
          <p>
            Budgeted sequential inspection console. Choose a project, policy, and budget — SENTINEL
            reveals factors in VoI order and updates P(severe).
          </p>
        </div>
      </header>

      <section className="panel" style={{ marginBottom: '1rem' }}>
        <div className="toolbar" style={{ marginBottom: 0 }}>
          <div className="field" style={{ minWidth: 180 }}>
            <label>Project PID</label>
            <select
              value={pid}
              onChange={(e) => setPid(e.target.value)}
            >
              <option value="">Select…</option>
              {pid && !pids.includes(pid) && <option value={pid}>{pid}</option>}
              {pids.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Policy</label>
            <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
              {POLICIES.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Budget B</label>
            <select value={budget} onChange={(e) => setBudget(Number(e.target.value))}>
              {[1, 2, 3, 4].map((b) => (
                <option key={b} value={b}>
                  {b}
                </option>
              ))}
            </select>
          </div>
          <button className="btn btn-primary" disabled={!pid || busy} onClick={run}>
            {busy ? 'Running…' : 'Run inspection'}
          </button>
        </div>
      </section>

      {err && <div className="error">{err}</div>}

      {!result && !err && (
        <div className="empty">Select a test project and run an inspection to see the live path.</div>
      )}

      {result && (
        <div className="grid grid-2">
          <section className="panel">
            <h2>Risk state</h2>
            <div className="risk-wrap">
              <div className="gauge" style={{ ['--p' as string]: p * 100 }}>
                <div className="gauge-inner">
                  <strong>{pct(p, 1)}</strong>
                  <span style={{ fontSize: '0.78rem', color: 'var(--slate)' }}>P(severe)</span>
                </div>
              </div>
              <div>
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.75rem' }}>
                  {result.alerted || step?.alert ? (
                    <span className="badge badge-danger">ALERT</span>
                  ) : (
                    <span className="badge badge-ok">HOLD</span>
                  )}
                  {result.severe_delay ? (
                    <span className="badge badge-warn">truth: severe</span>
                  ) : (
                    <span className="badge badge-muted">truth: normal</span>
                  )}
                  {result.correct ? (
                    <span className="badge badge-ok">decision matches label</span>
                  ) : (
                    <span className="badge badge-danger">decision mismatch</span>
                  )}
                </div>
                <p style={{ margin: 0, color: 'var(--slate)', fontSize: '0.9rem' }}>
                  Threshold τ = {num(result.threshold, 3)} · policy <strong>{result.policy}</strong> ·
                  budget {result.budget}
                </p>
                {result.champion && (
                  <p style={{ margin: '0.55rem 0 0', fontSize: '0.9rem' }}>
                    Early-warning ML: P = {pct(result.champion.p_severe)} (τ ={' '}
                    {num(result.champion.threshold, 2)}){' '}
                    {result.champion.alert ? (
                      <span className="badge badge-danger">flag</span>
                    ) : (
                      <span className="badge badge-ok">clear</span>
                    )}
                  </p>
                )}
                <div style={{ marginTop: '0.9rem', display: 'flex', gap: '0.5rem' }}>
                  <button
                    className="btn btn-ghost"
                    disabled={cursor <= 0}
                    onClick={() => setCursor((c) => Math.max(0, c - 1))}
                  >
                    Prev step
                  </button>
                  <button
                    className="btn btn-ghost"
                    disabled={!result || cursor >= result.steps.length - 1}
                    onClick={() => setCursor((c) => Math.min(result.steps.length - 1, c + 1))}
                  >
                    Next step
                  </button>
                </div>
              </div>
            </div>

            <h2 style={{ marginTop: '1.25rem' }}>Inspection timeline</h2>
            <div className="timeline">
              {result.steps.map((s, i) => (
                <div key={s.step} className={`step-row ${i === cursor ? 'active' : ''}`}>
                  <div className="n">{s.step}</div>
                  <div>
                    <strong>
                      {s.action === 'init'
                        ? 'Initialize evidence'
                        : `Inspect ${s.factor_label || s.factor}`}
                    </strong>
                    <div style={{ color: 'var(--slate)', fontSize: '0.82rem' }}>
                      {s.action === 'inspect' ? 'Observed present' : 'Clamped unmentioned factors absent'}
                    </div>
                  </div>
                  <div style={{ fontFamily: 'var(--font-brand)', fontWeight: 700 }}>{pct(s.p_severe)}</div>
                </div>
              ))}
            </div>
          </section>

          <section className="panel">
            <h2>Factor board</h2>
            <div className="factor-board">
              {factors.map((f) => {
                const present = result.present_factors.includes(f.id)
                const seen = inspected.has(f.id)
                const cls = seen ? 'inspected' : present ? 'present' : 'absent'
                return (
                  <div key={f.id} className={`factor-chip ${cls}`}>
                    <div className="fid">{f.id}</div>
                    <div className="flabel">{f.label}</div>
                    <div style={{ marginTop: '0.35rem', fontSize: '0.75rem', color: 'var(--slate)' }}>
                      {seen ? 'inspected' : present ? 'available' : 'known absent'}
                    </div>
                  </div>
                )
              })}
            </div>

            <h2 style={{ marginTop: '1.25rem' }}>Baseline first picks</h2>
            <table className="data">
              <thead>
                <tr>
                  <th>Policy</th>
                  <th>First factor</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(result.baselines_first_pick).map(([k, v]) => (
                  <tr key={k}>
                    <td>{k}</td>
                    <td>{v || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </div>
      )}
    </>
  )
}
