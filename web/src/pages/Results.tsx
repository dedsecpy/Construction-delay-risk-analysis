import { useEffect, useState } from 'react'
import { api, pct, num } from '../api'
import type { Overview } from '../api'

export function Results() {
  const [data, setData] = useState<Overview | null>(null)

  useEffect(() => {
    api.overview().then(setData)
  }, [])

  if (!data) return <div className="loading">Loading results…</div>
  const c = data.champion

  return (
    <>
      <header className="page-head">
        <div>
          <h1>Benchmarks</h1>
          <p>
            DRIN scorecard — early-warning champion metrics and literature comparison for the
            budgeted VoI delay-risk study.
          </p>
        </div>
      </header>

      <div className="grid grid-3" style={{ marginBottom: '1rem' }}>
        <div className="kpi">
          <div className="label">Accuracy</div>
          <div className="value">{pct(c.accuracy as number)}</div>
        </div>
        <div className="kpi">
          <div className="label">Precision / Recall</div>
          <div className="value" style={{ fontSize: '1.35rem' }}>
            {pct(c.precision as number)} / {pct(c.recall as number)}
          </div>
        </div>
        <div className="kpi">
          <div className="label">Cohen κ</div>
          <div className="value">{num(c.kappa as number, 3)}</div>
        </div>
      </div>

      <div className="grid grid-2">
        <section className="panel">
          <h2>Champion configuration</h2>
          <table className="data">
            <tbody>
              <tr>
                <td>Model</td>
                <td>{String(c.model ?? '—')}</td>
              </tr>
              <tr>
                <td>Early fraction</td>
                <td>{String(c.early_frac ?? '—')}</td>
              </tr>
              <tr>
                <td>Alert threshold</td>
                <td>{num(c.threshold as number, 3)}</td>
              </tr>
              <tr>
                <td>F1</td>
                <td>{num(c.f1 as number, 3)}</td>
              </tr>
              <tr>
                <td>ROC-AUC</td>
                <td>{num(c.roc_auc as number, 3)}</td>
              </tr>
            </tbody>
          </table>
        </section>

        <section className="panel">
          <h2>Versus literature</h2>
          <table className="data">
            <thead>
              <tr>
                <th>Benchmark</th>
                <th>Published</th>
                <th>DRIN</th>
                <th>Beat</th>
              </tr>
            </thead>
            <tbody>
              {data.literature.map((r) => (
                <tr key={r.literature_metric}>
                  <td>{r.literature_metric.replaceAll('_', ' ')}</td>
                  <td>{pct(r.literature_value)}</td>
                  <td>{pct(r.our_test_accuracy)}</td>
                  <td>
                    {r.beat ? (
                      <span className="badge badge-ok">yes</span>
                    ) : (
                      <span className="badge badge-danger">no</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </>
  )
}
