import { useEffect } from 'react'
import { useStore } from '../store'
import { getSessions, deleteSession, getCollections, getProviders } from '../api'
import './Sidebar.css'

function formatDate(iso: string) {
  const d = new Date(iso)
  const now = new Date()
  const diff = now.getTime() - d.getTime()
  if (diff < 86400000) return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  if (diff < 604800000) return d.toLocaleDateString([], { weekday: 'short' })
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

export function Sidebar() {
  const sessions = useStore(s => s.sessions)
  const activeSessionId = useStore(s => s.activeSessionId)
  const providers = useStore(s => s.providers)
  const activeProvider = useStore(s => s.activeProvider)
  const sidebarView = useStore(s => s.sidebarView)
  const setSessions = useStore(s => s.setSessions)
  const setActiveSessionId = useStore(s => s.setActiveSessionId)
  const setCollections = useStore(s => s.setCollections)
  const setProviders = useStore(s => s.setProviders)
  const setActiveProvider = useStore(s => s.setActiveProvider)
  const setSidebarView = useStore(s => s.setSidebarView)

  useEffect(() => {
    Promise.all([getSessions(), getCollections(), getProviders()]).then(([s, c, p]) => {
      setSessions(s)
      setCollections(c)
      setProviders(p.providers)
      setActiveProvider(p.active)
    })
  }, [setSessions, setCollections, setProviders, setActiveProvider])

  function newChat() {
    setActiveSessionId(null)
    setSidebarView('chat')
  }

  async function removeSession(id: string, e: React.MouseEvent) {
    e.stopPropagation()
    await deleteSession(id)
    setSessions(sessions.filter(s => s.id !== id))
    if (activeSessionId === id) setActiveSessionId(null)
  }

  const activeProviderInfo = providers.find(p => p.name === activeProvider)

  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-icon">&#9670;</span>
        <span className="brand-name">Lore</span>
      </div>

      <div className="nav">
        <button className={`nav-btn ${sidebarView === 'chat' ? 'active' : ''}`} onClick={() => setSidebarView('chat')}>
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M1 2a1 1 0 011-1h10a1 1 0 011 1v7a1 1 0 01-1 1H8l-3 3V10H2a1 1 0 01-1-1V2z" stroke="currentColor" strokeWidth="1.4"/>
          </svg>
          Chat
        </button>
        <button className={`nav-btn ${sidebarView === 'ingest' ? 'active' : ''}`} onClick={() => setSidebarView('ingest')}>
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M7 1v8M4 6l3 3 3-3M2 11h10" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Add content
        </button>
        <button className={`nav-btn ${sidebarView === 'library' ? 'active' : ''}`} onClick={() => setSidebarView('library')}>
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <rect x="1" y="1" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.4"/>
            <rect x="8" y="1" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.4"/>
            <rect x="1" y="8" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.4"/>
            <rect x="8" y="8" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.4"/>
          </svg>
          Library
        </button>
      </div>

      <div className="content">
        <button className="new-chat-btn" onClick={newChat}>
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path d="M6 1v10M1 6h10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
          </svg>
          New chat
        </button>

        {sessions.length > 0 && (
          <>
            <div className="section-label">Recent</div>
            <div className="session-list">
              {sessions.map(session => (
                <div
                  key={session.id}
                  className={`session-item ${activeSessionId === session.id && sidebarView === 'chat' ? 'active' : ''}`}
                  role="button"
                  tabIndex={0}
                  onClick={() => { setActiveSessionId(session.id); setSidebarView('chat') }}
                  onKeyDown={e => e.key === 'Enter' && setActiveSessionId(session.id)}
                >
                  <span className="session-title">{session.title}</span>
                  <div className="session-meta">
                    <span className="session-date">{formatDate(session.updated_at)}</span>
                    <button
                      className="del-btn"
                      onClick={e => removeSession(session.id, e)}
                      title="Delete"
                      aria-label="Delete session"
                    >
                      <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                        <path d="M1 1l8 8M9 1L1 9" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
                      </svg>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      <div className="footer">
        {activeProviderInfo ? (
          <div className="provider-badge">
            <span className={`provider-dot ${activeProviderInfo.authenticated ? 'ok' : ''}`} />
            <span className="provider-name">{activeProviderInfo.display_name}</span>
          </div>
        ) : (
          <div className="provider-badge">
            <span className="provider-dot" />
            <span className="provider-name" style={{ color: '#555' }}>No provider</span>
          </div>
        )}
      </div>
    </aside>
  )
}
