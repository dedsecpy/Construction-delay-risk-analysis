import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, pct } from '../api'
import type { Overview } from '../api'

export function Command() {
  const [data, setData] = useState<Overview | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    api.overview().then(setData).catch((e) => setErr(String(e.message || e)))
  }, [])

  if (err) return <div className="error">{err}</div>
  if (!data) return <div className="loading">Loading command center…</div>

  const c = data.champion

  return (
    <>
      <header className="page-head">
        <div>
          <h1>Command</h1>
          <p>
            Live operational view of SENTINEL — early-warning champion metrics, literature
            position, and budgeted inspection scoreboard.
          </p>
        </div>
        <Link className="btn btn-primary" to="/">
          Open analysis lab
        </Link>
      </header>

      <div className="grid grid-4" style={{ marginBottom: '1rem' }}>
        <div className="kpi">
          <div className="label">Test accuracy</div>
          <div className="value">{pct(c.accuracy as number)}</div>
          <div className="hint">Early-warning champion</div>
        </div>
        <div className="kpi">
          <div className="label">ROC-AUC</div>
          <div className="value">{pct(c.roc_auc as number)}</div>
          <div className="hint">Held-out TEST PIDs</div>
        </div>
        <div className="kpi">
          <div className="label">Projects</div>
          <div className="value">{data.counts.projects.toLocaleString()}</div>
          <div className="hint">
            {data.counts.train}/{data.counts.val}/{data.counts.test} split
          </div>
        </div>
        <div className="kpi">
          <div className="label">Literature</div>
          <div className="value" style={{ fontSize: '1.25rem', marginTop: '0.7rem' }}>
            {data.beats.tagharobi_89 ? (
              <span className="badge badge-ok">Beats 0.89</span>
            ) : (
              <span className="badge badge-warn">Below 0.89</span>
            )}
          </div>
          <div className="hint">
            {data.beats.khodabakhshian_85 ? 'Also beats Khodabakhshian 0.85' : '—'}
          </div>
        </div>
      </div>

      <div className="grid grid-2">
        <section className="panel">
          <h2>Literature head-to-head</h2>
          {data.literature.map((row) => {
            const ours = row.our_test_accuracy
            const theirs = row.literature_value
            return (
              <div className="lit-bar" key={row.literature_metric}>
                <div style={{ fontSize: '0.85rem' }}>
                  {row.literature_metric.replaceAll('_', ' ')}
                  <div style={{ color: 'var(--slate)', fontSize: '0.75rem' }}>
                    theirs {pct(theirs)} · ours {pct(ours)}
                  </div>
                </div>
                <div className="lit-track">
                  <div className="lit-fill" style={{ width: `${Math.min(100, ours * 100)}%` }} />
                </div>
                <div>
                  {row.beat ? (
                    <span className="badge badge-ok">+{pct(row.margin_accuracy)}</span>
                  ) : (
                    <span className="badge badge-danger">miss</span>
                  )}
                </div>
              </div>
            )
          })}
        </section>

        <section className="panel">
          <h2>Policy scoreboard · B = 2</h2>
          <table className="data">
            <thead>
              <tr>
                <th>Policy</th>
                <th>Recall</th>
                <th>Precision</th>
                <th>FPR</th>
                <th>AUC</th>
              </tr>
            </thead>
            <tbody>
              {data.scoreboard_b2.map((r) => (
                <tr key={String(r.policy)}>
                  <td>{String(r.policy)}</td>
                  <td>{pct(Number(r.recall))}</td>
                  <td>{pct(Number(r.precision))}</td>
                  <td>{pct(Number(r.fpr))}</td>
                  <td>{pct(Number(r.roc_auc))}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ margin: '0.9rem 0 0', color: 'var(--slate)', fontSize: '0.85rem' }}>
            Noisy-OR p_cause = {data.learned_cpts.p_cause ?? '—'} · leak ={' '}
            {data.learned_cpts.leak ?? '—'}
          </p>
        </section>
      </div>
    </>
  )
}
