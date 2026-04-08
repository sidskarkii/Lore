import { useState } from 'react'
import { useStore } from '../store'
import { ingestFile, ingestUrl, ingestYouTube, ingestFolder, getCollections } from '../api'
import type { IngestJob } from '../types'
import './IngestView.css'

type SourceType = 'file' | 'url' | 'youtube' | 'folder'

const SOURCE_TYPES: { id: SourceType; label: string; icon: string; desc: string }[] = [
  { id: 'file', label: 'File', icon: '📄', desc: 'PDF, EPUB, Markdown, Code, Audio' },
  { id: 'url', label: 'Web Page', icon: '🌐', desc: 'Articles, docs, blog posts' },
  { id: 'youtube', label: 'YouTube', icon: '▶', desc: 'Videos and playlists' },
  { id: 'folder', label: 'Folder', icon: '📁', desc: 'Batch video/audio files' },
]

export function IngestView() {
  const [source, setSource] = useState<SourceType>('file')
  const [name, setName] = useState('')
  const [topic, setTopic] = useState('')
  const [subtopic, setSubtopic] = useState('')
  const [input, setInput] = useState('')
  const [jobs, setJobs] = useState<IngestJob[]>([])
  const [ingesting, setIngesting] = useState(false)

  const setCollections = useStore(s => s.setCollections)

  const placeholder: Record<SourceType, string> = {
    file: 'C:\\Books\\DDIA.pdf',
    url: 'https://docs.python.org/3/tutorial/',
    youtube: 'https://youtube.com/watch?v=...',
    folder: 'D:\\Courses\\Blender101',
  }

  const inputLabel: Record<SourceType, string> = {
    file: 'File path',
    url: 'URL',
    youtube: 'YouTube URL',
    folder: 'Folder path',
  }

  const canSubmit = !!name && !!input && !ingesting

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!canSubmit) return

    setIngesting(true)
    const job: IngestJob = { type: source, label: input, status: 'running', message: 'Processing...' }
    setJobs(prev => [job, ...prev])

    try {
      let res: any
      switch (source) {
        case 'file':
          res = await ingestFile({ path: input, name, topic, subtopic })
          break
        case 'url':
          res = await ingestUrl({ url: input, name, topic, subtopic })
          break
        case 'youtube':
          res = await ingestYouTube({ url: input, name, topic, subtopic })
          break
        case 'folder':
          res = await ingestFolder({ path: input, name, topic, subtopic })
          break
      }

      if (res.success) {
        job.status = 'done'
        job.message = `${res.chunks} chunks indexed`
        const updated = await getCollections()
        setCollections(updated)
        setInput('')
        setName('')
      } else {
        job.status = 'error'
        job.message = res.error ?? 'Unknown error'
      }
    } catch (err: any) {
      job.status = 'error'
      job.message = err.message ?? 'Request failed'
    }

    setIngesting(false)
    setJobs(prev => [...prev])
  }

  return (
    <div className="ingest-view">
      <div className="ingest-header">
        <h1>Add content</h1>
        <p>Ingest knowledge from any source into your vault.</p>
      </div>

      <div className="source-grid">
        {SOURCE_TYPES.map(s => (
          <button
            key={s.id}
            className={`source-card ${source === s.id ? 'active' : ''}`}
            onClick={() => { setSource(s.id); setInput('') }}
          >
            <span className="source-icon">{s.icon}</span>
            <span className="source-label">{s.label}</span>
            <span className="source-desc">{s.desc}</span>
          </button>
        ))}
      </div>

      <form className="ingest-form" onSubmit={handleSubmit}>
        <label className="form-field">
          <span className="form-label">{inputLabel[source]}</span>
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder={placeholder[source]}
            type={source === 'url' || source === 'youtube' ? 'url' : 'text'}
            disabled={ingesting}
          />
        </label>

        <label className="form-field">
          <span className="form-label">Collection name</span>
          <input
            value={name}
            onChange={e => setName(e.target.value)}
            placeholder="e.g. DDIA, React Docs, Blender Course"
            disabled={ingesting}
          />
        </label>

        <div className="form-row">
          <label className="form-field">
            <span className="form-label">Topic</span>
            <input
              value={topic}
              onChange={e => setTopic(e.target.value)}
              placeholder="engineering"
              disabled={ingesting}
            />
          </label>
          <label className="form-field">
            <span className="form-label">Subtopic</span>
            <input
              value={subtopic}
              onChange={e => setSubtopic(e.target.value)}
              placeholder="databases"
              disabled={ingesting}
            />
          </label>
        </div>

        <button className="ingest-submit" type="submit" disabled={!canSubmit}>
          {ingesting ? 'Ingesting...' : 'Start ingestion'}
        </button>
      </form>

      {jobs.length > 0 && (
        <div className="jobs-section">
          <h3>Activity</h3>
          <div className="jobs-list">
            {jobs.map((job, i) => (
              <div key={i} className={`job-item ${job.status}`}>
                <div className={`job-dot ${job.status}`} />
                <div className="job-detail">
                  <span className="job-path">{job.label}</span>
                  <span className="job-status">{job.message}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
