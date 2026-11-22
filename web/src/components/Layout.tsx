import { NavLink, Outlet } from 'react-router-dom'

const links = [
  { to: '/', label: 'Portfolio' },
  { to: '/analyze', label: 'Project review' },
]

export function Layout() {
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">
            DR<em>IN</em>
          </span>
          <span className="brand-sub">Schedule Risk Operations</span>
        </div>
        <nav className="nav">
          {links.map((l) => (
            <NavLink key={l.to} to={l.to} end={l.to === '/'}>
              {l.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="page">
        <Outlet />
      </main>
    </div>
  )
}
