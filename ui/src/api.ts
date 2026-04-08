import type { SearchResult, Session, Collection, ProviderInfo } from './types'

const BASE = 'http://localhost:8000'

// ── Chat ─────────────────────────────────────────────────────────────────────

export interface StreamCallbacks {
  onSource?: (sources: SearchResult[]) => void
  onToken?: (token: string) => void
  onStatus?: (msg: string) => void
  onSession?: (id: string) => void
  onDone?: () => void
  onError?: (err: string) => void
}

export async function streamChat(
  messages: { role: string; content: string }[],
  sessionId: string | null,
  opts: { multiHop?: boolean; topic?: string; subtopic?: string } = {},
  callbacks: StreamCallbacks
): Promise<void> {
  const res = await fetch(`${BASE}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages,
      session_id: sessionId,
      multi_hop: opts.multiHop ?? false,
      topic: opts.topic ?? null,
      subtopic: opts.subtopic ?? null,
      n_sources: 5,
    }),
  })

  if (!res.ok) {
    callbacks.onError?.(`HTTP ${res.status}`)
    return
  }

  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  let doneFired = false

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    const chunk = decoder.decode(value, { stream: true })
    buf += chunk

    if (buf.length < 500) console.log('[SSE raw]', JSON.stringify(buf))

    const parts = buf.split('\n\n')
    buf = parts.pop() ?? ''

    for (const part of parts) {
      const lines = part.trim().split('\n')
      let eventType = 'message'
      let dataLine = ''

      for (const line of lines) {
        if (line.startsWith('event:')) eventType = line.slice(6).trim()
        if (line.startsWith('data:')) dataLine = line.slice(5).trim()
      }

      if (!dataLine) continue

      console.log('[SSE]', eventType, dataLine.slice(0, 30))

      try {
        const data = JSON.parse(dataLine)
        if (eventType === 'source') callbacks.onSource?.([data])
        else if (eventType === 'token') callbacks.onToken?.(typeof data === 'string' ? data : dataLine)
        else if (eventType === 'status') callbacks.onStatus?.(data)
        else if (eventType === 'session') callbacks.onSession?.(typeof data === 'string' ? data : String(data))
        else if (eventType === 'done') { doneFired = true; callbacks.onDone?.() }
        else if (eventType === 'error') callbacks.onError?.(typeof data === 'string' ? data : JSON.stringify(data))
      } catch {
        if (eventType === 'token') callbacks.onToken?.(dataLine)
        else if (eventType === 'error') callbacks.onError?.(dataLine)
        else if (eventType === 'session') callbacks.onSession?.(dataLine)
        else if (eventType === 'done') { doneFired = true; callbacks.onDone?.() }
      }
    }
  }

  if (!doneFired) callbacks.onDone?.()
}

// ── Sessions ──────────────────────────────────────────────────────────────────

export async function getSessions(): Promise<Session[]> {
  const r = await fetch(`${BASE}/api/sessions`)
  const d = await r.json()
  return d.sessions ?? []
}

export async function getSession(id: string) {
  const r = await fetch(`${BASE}/api/sessions/${id}`)
  return r.json()
}

export async function deleteSession(id: string): Promise<void> {
  await fetch(`${BASE}/api/sessions/${id}`, { method: 'DELETE' })
}

export async function renameSession(id: string, title: string): Promise<void> {
  await fetch(`${BASE}/api/sessions/${id}/rename`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  })
}

// ── Collections ───────────────────────────────────────────────────────────────

export async function getCollections(): Promise<Collection[]> {
  const r = await fetch(`${BASE}/api/collections`)
  const d = await r.json()
  return d.collections ?? []
}

// ── Providers ─────────────────────────────────────────────────────────────────

export async function getProviders(): Promise<{ providers: ProviderInfo[]; active: string | null }> {
  const r = await fetch(`${BASE}/api/providers`)
  return r.json()
}

export async function setActiveProvider(provider: string): Promise<void> {
  await fetch(`${BASE}/api/providers/active`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider }),
  })
}

// ── Ingest ────────────────────────────────────────────────────────────────────

export async function ingestFolder(payload: {
  path: string; name: string; topic: string; subtopic: string
}) {
  const r = await fetch(`${BASE}/api/ingest/folder`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...payload, language: 'en' }),
  })
  return r.json()
}

export async function ingestYouTube(payload: {
  url: string; name: string; topic: string; subtopic: string
}) {
  const r = await fetch(`${BASE}/api/ingest/youtube`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...payload, language: 'en' }),
  })
  return r.json()
}

export async function ingestFile(payload: {
  path: string; name: string; topic: string; subtopic: string; source_type?: string
}) {
  const r = await fetch(`${BASE}/api/ingest/file`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return r.json()
}

export async function ingestUrl(payload: {
  url: string; name: string; topic: string; subtopic: string
}) {
  const r = await fetch(`${BASE}/api/ingest/url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return r.json()
}

export async function deleteCollection(collection: string): Promise<void> {
  await fetch(`${BASE}/api/collections`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ collection }),
  })
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function getHealth() {
  const r = await fetch(`${BASE}/api/health`)
  return r.json()
}
