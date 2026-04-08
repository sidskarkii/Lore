import { useState } from 'react'
import { useStore } from '../store'
import { ingestFolder, ingestYouTube, ingestFile, ingestUrl, getCollections } from '../api'
import type { IngestJob } from '../types'
import './IngestPanel.css'

type Tab = 'youtube' | 'file' | 'url' | 'folder'

export function IngestPanel() {
  const [tab, setTab] = useState<Tab>('file')
  const [name, setName] = useState('')
  const [topic, setTopic] = useState('')
  const [subtopic, setSubtopic] = useState('')
  const [filePath, setFilePath] = useState('')
  const [folderPath, setFolderPath] = useState('')
  const [ytUrl, setYtUrl] = useState('')
  const [webUrl, setWebUrl] = useState('')
  const [jobs, setJobs] = useState<IngestJob[]>([])

  const setCollections = useStore(s => s.setCollections)

  const canSubmit = !!name && (
    tab === 'file' ? !!filePath :
    tab === 'folder' ? !!folderPath :
    tab === 'youtube' ? !!ytUrl :
    tab === 'url' ? !!webUrl : false
  )

  const inputLabel = () => {
    switch (tab) {
      case 'file': return filePath
      case 'folder': return folderPath
      case 'youtube': return ytUrl
      case 'url': return webUrl
    }
  }

  async function runIngest(e: React.FormEvent) {
    e.preventDefault()
    const label = inputLabel()
    if (!label || !name) return

    const job: IngestJob = { type: tab, label, status: 'running', message: 'Ingesting...' }
    setJobs(prev => [job, ...prev])

    try {
      let res: any
      switch (tab) {
        case 'file':
          res = await ingestFile({ path: filePath, name, topic, subtopic })
          break
        case 'folder':
          res = await ingestFolder({ path: folderPath, name, topic, subtopic })
          break
        case 'youtube':
          res = await ingestYouTube({ url: ytUrl, name, topic, subtopic })
          break
        case 'url':
          res = await ingestUrl({ url: webUrl, name, topic, subtopic })
          break
      }

      if (res.success) {
        job.status = 'done'
        job.message = `Done — ${res.chunks} chunks indexed`
        const updated = await getCollections()
        setCollections(updated)
      } else {
        job.status = 'error'
        job.message = res.error ?? 'Unknown error'
      }
    } catch (err: any) {
      job.status = 'error'
      job.message = err.message ?? 'Request failed'
    }

    setJobs(prev => [...prev])
  }

  return (
    <div className="ingest">
      <div className="section-header">Add content</div>

      <div className="tabs">
        <button className={`tab ${tab === 'file' ? 'active' : ''}`} onClick={() => setTab('file')}>
          File
        </button>
        <button className={`tab ${tab === 'url' ? 'active' : ''}`} onClick={() => setTab('url')}>
          Web
        </button>
        <button className={`tab ${tab === 'youtube' ? 'active' : ''}`} onClick={() => setTab('youtube')}>
          YouTube
        </button>
        <button className={`tab ${tab === 'folder' ? 'active' : ''}`} onClick={() => setTab('folder')}>
          Folder
        </button>
      </div>

      <form className="form" onSubmit={runIngest}>
        {tab === 'file' && (
          <label className="field">
            <span>File path</span>
            <input value={filePath} onChange={e => setFilePath(e.target.value)}
              placeholder="C:\Books\DDIA.pdf or .epub, .md, .py, .mp3" />
          </label>
        )}
        {tab === 'folder' && (
          <label className="field">
            <span>Folder path</span>
            <input value={folderPath} onChange={e => setFolderPath(e.target.value)}
              placeholder="C:\Videos\BlenderTuts" />
          </label>
        )}
        {tab === 'youtube' && (
          <label className="field">
            <span>YouTube URL</span>
            <input value={ytUrl} onChange={e => setYtUrl(e.target.value)}
              placeholder="https://youtube.com/watch?v=..." type="url" />
          </label>
        )}
        {tab === 'url' && (
          <label className="field">
            <span>Web URL</span>
            <input value={webUrl} onChange={e => setWebUrl(e.target.value)}
              placeholder="https://docs.python.org/3/tutorial/" type="url" />
          </label>
        )}

        <label className="field">
          <span>Collection name</span>
          <input value={name} onChange={e => setName(e.target.value)} placeholder="DDIA, React Docs, etc." />
        </label>

        <div className="row">
          <label className="field">
            <span>Topic</span>
            <input value={topic} onChange={e => setTopic(e.target.value)} placeholder="engineering" />
          </label>
          <label className="field">
            <span>Subtopic</span>
            <input value={subtopic} onChange={e => setSubtopic(e.target.value)} placeholder="databases" />
          </label>
        </div>

        {tab === 'file' && (
          <div className="hint">
            Supports: PDF, EPUB, Markdown, plain text, code files, audio
          </div>
        )}

        <button className="submit-btn" type="submit" disabled={!canSubmit}>
          {tab === 'url' ? 'Fetch & ingest' : 'Start ingestion'}
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
