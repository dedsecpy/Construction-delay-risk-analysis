import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import type { ProjectRow } from '../api'

export function Projects() {
  const nav = useNavigate()
  const [q, setQ] = useState('')
  const [split, setSplit] = useState('test')
  const [severe, setSevere] = useState<string>('all')
  const [rows, setRows] = useState<ProjectRow[]>([])
  const [total, setTotal] = useState(0)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    const params = new URLSearchParams({ limit: '60', offset: '0' })
    if (q) params.set('q', q)
    if (split) params.set('split', split)
    if (severe !== 'all') params.set('severe', severe)
    api
      .projects(params)
      .then((r) => {
        setRows(r.items)
        setTotal(r.total)
      })
      .catch((e) => setErr(String(e.message || e)))
  }, [q, split, severe])

  return (
    <>
      <header className="page-head">
        <div>
          <h1>Projects</h1>
          <p>Browse NYC capital-project episodes. Open any PID in the inspection console.</p>
        </div>
      </header>

      <div className="toolbar">
        <div className="field">
          <label>Search PID</label>
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. 850" />
        </div>
        <div className="field">
          <label>Split</label>
          <select value={split} onChange={(e) => setSplit(e.target.value)}>
            <option value="">All</option>
            <option value="train">Train</option>
            <option value="val">Val</option>
            <option value="test">Test</option>
          </select>
        </div>
        <div className="field">
          <label>Severe</label>
          <select value={severe} onChange={(e) => setSevere(e.target.value)}>
            <option value="all">All</option>
            <option value="1">Severe only</option>
            <option value="0">Non-severe</option>
          </select>
        </div>
      </div>

      {err && <div className="error">{err}</div>}

      <section className="panel">
        <h2>
          Queue <span className="badge badge-muted">{total} projects</span>
        </h2>
        <table className="data">
          <thead>
            <tr>
              <th>PID</th>
              <th>Agency</th>
              <th>Phase</th>
              <th>Factors</th>
              <th>Updates</th>
              <th>Label</th>
              <th>Split</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.pid} onClick={() => nav(`/inspect?pid=${encodeURIComponent(r.pid)}`)}>
                <td>
                  <strong>{r.pid}</strong>
                </td>
                <td>{r.managing_agency || '—'}</td>
                <td>{r.phase || '—'}</td>
                <td>{r.n_factors}</td>
                <td>{r.n_updates}</td>
                <td>
                  {r.severe_delay ? (
                    <span className="badge badge-danger">severe</span>
                  ) : (
                    <span className="badge badge-ok">normal</span>
                  )}
                </td>
                <td>{r.split}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  )
}
