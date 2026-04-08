import { useEffect, useState } from 'react'
import { useStore } from '../store'
import { getCollections, deleteCollection, getHealth } from '../api'
import './LibraryView.css'

export function LibraryView() {
  const collections = useStore(s => s.collections)
  const setCollections = useStore(s => s.setCollections)
  const [totalChunks, setTotalChunks] = useState(0)

  useEffect(() => {
    getCollections().then(setCollections)
    getHealth().then(h => setTotalChunks(h.total_chunks ?? 0))
  }, [setCollections])

  async function handleDelete(collection: string) {
    await deleteCollection(collection)
    const updated = await getCollections()
    setCollections(updated)
    const h = await getHealth()
    setTotalChunks(h.total_chunks ?? 0)
  }

  return (
    <div className="library-view">
      <div className="library-header">
        <div>
          <h1>Library</h1>
          <p>Your indexed knowledge base.</p>
        </div>
        <div className="library-stats">
          <div className="stat">
            <span className="stat-value">{collections.length}</span>
            <span className="stat-label">Collections</span>
          </div>
          <div className="stat">
            <span className="stat-value">{totalChunks.toLocaleString()}</span>
            <span className="stat-label">Chunks</span>
          </div>
        </div>
      </div>

      {collections.length > 0 ? (
        <div className="collection-grid">
          {collections.map(col => (
            <div key={col.collection} className="lib-card">
              <div className="lib-card-top">
                <div className="lib-card-title">{col.collection_display}</div>
                <button
                  className="lib-card-del"
                  title="Delete collection"
                  onClick={() => handleDelete(col.collection)}
                >
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                    <path d="M2 2l8 8M10 2L2 10" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
                  </svg>
                </button>
              </div>
              <div className="lib-card-tags">
                {col.topic && <span className="lib-tag">{col.topic}</span>}
                {col.subtopic && <span className="lib-tag">{col.subtopic}</span>}
              </div>
              <div className="lib-card-footer">
                <span className="lib-ep-count">{col.episode_count} episode{col.episode_count !== 1 ? 's' : ''}</span>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="library-empty">
          <div className="empty-icon">&#9671;</div>
          <p>No content indexed yet.</p>
          <p className="empty-sub">Add content from the sidebar to get started.</p>
        </div>
      )}
    </div>
  )
}
