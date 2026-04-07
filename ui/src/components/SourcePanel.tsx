import type { SearchResult } from '../types'
import './SourcePanel.css'

interface Props {
  open: boolean
  sources: SearchResult[]
  onClose: () => void
}

export function SourcePanel({ open, sources, onClose }: Props) {
  return (
    <>
      {open && <button className="backdrop" onClick={onClose} aria-label="Close sources" />}

      <aside className={`panel ${open ? 'open' : ''}`}>
        <div className="panel-header">
          <span className="panel-title">Sources</span>
          <button className="close-btn" onClick={onClose}>&#10005;</button>
        </div>

        <div className="panel-body">
          {sources.map((src, i) => (
            <div key={i} className="source-card">
              <div className="source-meta">
                <span className="source-index">{i + 1}</span>
                <div className="source-info">
                  <span className="source-collection">{src.collection_display}</span>
                  <span className="source-episode">{src.episode_title}</span>
                </div>
                {src.url ? (
                  <a className="source-timestamp" href={src.url} target="_blank" rel="noopener noreferrer">
                    {src.timestamp}
                    <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                      <path d="M1 9L9 1M9 1H3M9 1V7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                    </svg>
                  </a>
                ) : (
                  <span className="source-timestamp plain">{src.timestamp}</span>
                )}
              </div>
              <p className="source-text">{src.text}</p>
            </div>
          ))}

          {sources.length === 0 && (
            <p className="empty">No sources for this message.</p>
          )}
        </div>
      </aside>
    </>
  )
}
