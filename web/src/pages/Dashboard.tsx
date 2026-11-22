import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api, pct } from '../api'
import type { Overview, ProjectOption } from '../api'

export function Dashboard() {
  const [data, setData] = useState<Overview | null>(null)
  const [projects, setProjects] = useState<ProjectOption[]>([])
  const [filter, setFilter] = useState<'all' | 'severe' | 'normal'>('severe')
  const [q, setQ] = useState('')
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([api.overview(), api.options('test')])
      .then(([o, p]) => {
        setData(o)
        setProjects(p.items)
      })
      .catch((e) => setErr(String(e.message || e)))
  }, [])

  const visible = useMemo(() => {
    let rows = projects
    if (filter === 'severe') rows = rows.filter((p) => p.severe_delay === 1)
    if (filter === 'normal') rows = rows.filter((p) => p.severe_delay === 0)
    if (q.trim()) {
      const s = q.toLowerCase()
      rows = rows.filter(
        (p) =>
          p.name.toLowerCase().includes(s) ||
          p.pid.includes(s) ||
          (p.agency || '').toLowerCase().includes(s),
      )
    }
    return rows.slice(0, 40)
  }, [projects, filter, q])

  const factors = useMemo(() => {
    const rows = [...(data?.factor_counts || [])].sort((a, b) => b.rows - a.rows)
    return rows.filter((f) => f.factor !== 'other_admin' || f.rows > 0)
  }, [data])

  if (err) return <div className="error">{err}</div>
  if (!data) return <div className="loading">Loading portfolio…</div>

  const severeN = data.counts.severe_projects ?? 0
  const atRiskPct = data.counts.severe_rate

  return (
    <>
      <div className={`status-strip ${atRiskPct > 0.15 ? 'at-risk' : ''}`}>
        <div>
          <span className="k">Portfolio</span>
          <span className="v">{data.counts.projects.toLocaleString()} projects</span>
        </div>
        <div className="sep" />
        <div>
          <span className="k">At-risk (severe)</span>
          <span className="v" style={{ color: 'var(--orange)' }}>
            {severeN.toLocaleString()} · {pct(atRiskPct)}
          </span>
        </div>
        <div className="sep" />
        <div>
          <span className="k">Active delay drivers</span>
          <span className="v">{(data.counts.nonempty_reason_rows || 0).toLocaleString()} records</span>
        </div>
      </div>

      <header className="ph">
        <div>
          <h1>Portfolio risk</h1>
          <p>
            Monitor schedule-delay exposure across capital projects. Open a project to run
            budgeted factor inspection.
          </p>
        </div>
        <Link className="btn btn-primary" to="/analyze">
          Open project review
        </Link>
      </header>

      <div className="metrics">
        <div className="metric risk">
          <div className="lbl">Severe schedule risk</div>
          <div className="num">{severeN.toLocaleString()}</div>
          <div className="sub">Top-quartile cumulative slip</div>
        </div>
        <div className="metric primary">
          <div className="lbl">Projects tracked</div>
          <div className="num">{data.counts.projects.toLocaleString()}</div>
          <div className="sub">
            Train/val/test {data.counts.train}/{data.counts.val}/{data.counts.test}
          </div>
        </div>
        <div className="metric">
          <div className="lbl">Early-warning hit rate</div>
          <div className="num">{pct(data.champion.accuracy as number, 0)}</div>
          <div className="sub">Model accuracy on held-out projects</div>
        </div>
        <div className="metric">
          <div className="lbl">Inspection model AUC</div>
          <div className="num">{pct(data.champion.roc_auc as number, 0)}</div>
          <div className="sub">Ranking quality for severe class</div>
        </div>
      </div>

      <div className="layout-2">
        <section className="panel">
          <h2>Project queue</h2>
          <div className="toolbar">
            <div className="field" style={{ minWidth: 200 }}>
              <label>Search</label>
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Name, PID, agency"
              />
            </div>
            <div className="field">
              <label>Status</label>
              <select value={filter} onChange={(e) => setFilter(e.target.value as typeof filter)}>
                <option value="severe">At risk only</option>
                <option value="all">All test projects</option>
                <option value="normal">On track</option>
              </select>
            </div>
          </div>
          <table className="ops">
            <thead>
              <tr>
                <th>Status</th>
                <th>Project</th>
                <th>Agency</th>
                <th>Phase</th>
                <th>Drivers</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {visible.map((p) => (
                <tr key={p.pid}>
                  <td>
                    {p.severe_delay ? (
                      <span className="tag tag-risk">At risk</span>
                    ) : (
                      <span className="tag tag-ok">On track</span>
                    )}
                  </td>
                  <td>
                    <div className="name">{p.name}</div>
                    <div className="meta">PID {p.pid}</div>
                  </td>
                  <td>{p.agency || '—'}</td>
                  <td>{p.phase || '—'}</td>
                  <td>{p.n_factors}</td>
                  <td>
                    <Link className="btn btn-ghost btn-sm" to={`/analyze?pid=${encodeURIComponent(p.pid)}`}>
                      Review
                    </Link>
                  </td>
                </tr>
              ))}
              {!visible.length && (
                <tr>
                  <td colSpan={6}>
                    <div className="empty">No projects match this filter.</div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </section>

        <section className="panel">
          <h2>Delay drivers</h2>
          <p className="panel-note">
            Mapped schedule-change reasons across the portfolio (excluding unclassified filler where
            possible).
          </p>
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={factors} layout="vertical" margin={{ left: 8, right: 12 }}>
                <CartesianGrid stroke="#e6ebf0" strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis
                  type="category"
                  dataKey="label"
                  width={118}
                  tick={{ fontSize: 11, fill: '#333' }}
                />
                <Tooltip />
                <Bar dataKey="rows" barSize={14}>
                  {factors.map((f) => (
                    <Cell
                      key={f.factor}
                      fill={f.rows >= 400 ? '#ff4026' : f.rows >= 100 ? '#0b5dd0' : '#7a8fa3'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{ marginTop: '0.85rem' }}>
            {(data.severe_watchlist || []).slice(0, 4).map((w) => (
              <div
                key={w.pid}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  gap: '0.5rem',
                  padding: '0.55rem 0',
                  borderBottom: '1px solid var(--line)',
                }}
              >
                <div>
                  <span className="tag tag-risk">At risk</span>
                  <div className="name" style={{ marginTop: '0.35rem' }}>
                    {w.name}
                  </div>
                  <div className="meta">PID {w.pid}</div>
                </div>
                <Link className="btn btn-accent btn-sm" to={`/analyze?pid=${w.pid}`}>
                  Inspect
                </Link>
              </div>
            ))}
          </div>
        </section>
      </div>
    </>
  )
}
