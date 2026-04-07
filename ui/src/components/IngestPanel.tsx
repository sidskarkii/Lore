import { useState } from 'react'
import { useStore } from '../store'
import { ingestFolder, ingestYouTube, getCollections } from '../api'
import type { IngestJob } from '../types'
import './IngestPanel.css'

export function IngestPanel() {
  const [tab, setTab] = useState<'folder' | 'youtube'>('youtube')
  const [name, setName] = useState('')
  const [topic, setTopic] = useState('')
  const [subtopic, setSubtopic] = useState('')
  const [folderPath, setFolderPath] = useState('')
  const [ytUrl, setYtUrl] = useState('')
  const [jobs, setJobs] = useState<IngestJob[]>([])

  const setCollections = useStore(s => s.setCollections)

  const canSubmit = !!name && (tab === 'folder' ? !!folderPath : !!ytUrl)

  async function runIngest(e: React.FormEvent) {
    e.preventDefault()
    const label = tab === 'folder' ? folderPath : ytUrl
    if (!label || !name) return

    const job: IngestJob = { type: tab, label, status: 'running', message: 'Ingesting...' }
    setJobs(prev => [job, ...prev])

    try {
      const res = tab === 'folder'
        ? await ingestFolder({ path: folderPath, name, topic, subtopic })
        : await ingestYouTube({ url: ytUrl, name, topic, subtopic })

      if (res.success) {
        job.status = 'done'
        job.message = `Done - ${res.chunks} chunks indexed`
        const updated = await getCollections()
        setCollections(updated)
      } else {
        job.status = 'error'
        job.message = res.error ?? 'Unknown error'
      }
    } catch (e: any) {
      job.status = 'error'
      job.message = e.message ?? 'Request failed'
    }

    setJobs(prev => [...prev])
  }

  return (
    <div className="ingest">
      <div className="section-header">Add content</div>

      <div className="tabs">
        <button className={`tab ${tab === 'youtube' ? 'active' : ''}`} onClick={() => setTab('youtube')}>
          YouTube
        </button>
        <button className={`tab ${tab === 'folder' ? 'active' : ''}`} onClick={() => setTab('folder')}>
          Folder
        </button>
      </div>

      <form className="form" onSubmit={runIngest}>
        {tab === 'youtube' ? (
          <label className="field">
            <span>URL</span>
            <input value={ytUrl} onChange={e => setYtUrl(e.target.value)} placeholder="https://youtube.com/watch?v=..." type="url" />
          </label>
        ) : (
          <label className="field">
            <span>Folder path</span>
            <input value={folderPath} onChange={e => setFolderPath(e.target.value)} placeholder="C:\Videos\BlenderTuts" />
          </label>
        )}

        <label className="field">
          <span>Collection name</span>
          <input value={name} onChange={e => setName(e.target.value)} placeholder="Blender Rigging Series" />
        </label>

        <div className="row">
          <label className="field">
            <span>Topic</span>
            <input value={topic} onChange={e => setTopic(e.target.value)} placeholder="3d" />
          </label>
          <label className="field">
            <span>Subtopic</span>
            <input value={subtopic} onChange={e => setSubtopic(e.target.value)} placeholder="blender" />
          </label>
        </div>

        <button className="submit-btn" type="submit" disabled={!canSubmit}>
          Start ingestion
        </button>
      </form>

      {jobs.length > 0 && (
        <div className="jobs">
          <div className="section-header" style={{ marginTop: 8 }}>Recent</div>
          {jobs.map((job, i) => (
            <div key={i} className={`job-row ${job.status}`}>
              <div className={`job-indicator ${job.status === 'running' ? 'running' : ''}`} />
              <div className="job-info">
                <span className="job-label">{job.label}</span>
                <span className="job-msg">{job.message}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
