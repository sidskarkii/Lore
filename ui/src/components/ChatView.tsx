import { useState, useRef, useEffect, useCallback } from 'react'
import { fetchEventSource } from '@microsoft/fetch-event-source'
import { useStore } from '../store'
import { getSessions, getSession } from '../api'
import type { Message, SearchResult } from '../types'
import { SourcePanel } from './SourcePanel'
import './ChatView.css'

const API_BASE = 'http://localhost:8000'

function renderMarkdown(text: string): string {
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/```[\w]*\n([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>')
}

export function ChatView() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [statusMsg, setStatusMsg] = useState('')
  const [multiHop, setMultiHop] = useState(false)
  const [sourcePanelOpen, setSourcePanelOpen] = useState(false)
  const [activeSources, setActiveSources] = useState<SearchResult[]>([])

  const scrollRef = useRef<HTMLDivElement>(null)
  const activeSessionId = useStore(s => s.activeSessionId)
  const setSessions = useStore(s => s.setSessions)
  const setActiveSessionId = useStore(s => s.setActiveSessionId)

  // Track whether we're currently streaming to prevent loadSession mid-stream
  const streamingRef = useRef(false)

  const scrollToBottom = useCallback(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [])

  // Load session when activeSessionId changes (sidebar click, etc.)
  useEffect(() => {
    if (streamingRef.current) return // don't load during streaming
    if (activeSessionId) {
      getSession(activeSessionId).then(data => {
        setMessages(
          (data.messages ?? [])
            .filter((m: any) => m.role !== 'system')
            .map((m: any) => ({
              id: m.id,
              role: m.role as 'user' | 'assistant',
              content: m.content,
              sources: m.sources ?? [],
            }))
        )
        setTimeout(scrollToBottom, 50)
      })
    } else {
      setMessages([])
    }
  }, [activeSessionId, scrollToBottom])

  // Auto-scroll on message changes
  useEffect(() => { setTimeout(scrollToBottom, 20) }, [messages, scrollToBottom])

  async function send() {
    const text = input.trim()
    if (!text || sending) return

    setInput('')
    setSending(true)
    setStatusMsg('')
    streamingRef.current = true

    const assistantId = crypto.randomUUID()
    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: text }
    const assistantMsg: Message = { id: assistantId, role: 'assistant', content: '', streaming: true }

    const history = messages.map(m => ({ role: m.role, content: m.content }))
    setMessages(prev => [...prev, userMsg, assistantMsg])

    let pendingSessionId: string | null = null

    await fetchEventSource(`${API_BASE}/api/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: [...history, { role: 'user', content: text }],
        session_id: activeSessionId,
        multi_hop: multiHop,
        n_sources: 5,
      }),
      onmessage(ev) {
        const { event, data } = ev
        if (!data) return

        if (event === 'token') {
          let token: string
          try { token = JSON.parse(data) } catch { token = data }
          setMessages(prev => prev.map(m =>
            m.id === assistantId ? { ...m, content: m.content + token } : m
          ))
        } else if (event === 'source') {
          try {
            const src = JSON.parse(data)
            setMessages(prev => prev.map(m =>
              m.id === assistantId
                ? { ...m, sources: [...(m.sources ?? []), src] }
                : m
            ))
          } catch { /* ignore */ }
        } else if (event === 'session') {
          let id: string
          try { id = JSON.parse(data) } catch { id = data }
          pendingSessionId = id
          getSessions().then(list => setSessions(list))
        } else if (event === 'status') {
          try { setStatusMsg(JSON.parse(data)) } catch { setStatusMsg(data) }
        } else if (event === 'done') {
          setMessages(prev => prev.map(m =>
            m.id === assistantId ? { ...m, streaming: false } : m
          ))
          setStatusMsg('')
          setSending(false)
          streamingRef.current = false
          if (pendingSessionId) setActiveSessionId(pendingSessionId)
        } else if (event === 'error') {
          let msg: string
          try { msg = JSON.parse(data) } catch { msg = data }
          setMessages(prev => prev.map(m =>
            m.id === assistantId
              ? { ...m, content: `Error: ${msg}`, streaming: false }
              : m
          ))
          setSending(false)
          streamingRef.current = false
          setStatusMsg('')
        }
      },
      onerror(err) {
        console.error('[SSE error]', err)
        setMessages(prev => prev.map(m =>
          m.id === assistantId
            ? { ...m, content: `Error: ${err}`, streaming: false }
            : m
        ))
        setSending(false)
        streamingRef.current = false
        throw err // stop retrying
      },
      openWhenHidden: true,
    })
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="chat-wrap">
      <div className="messages" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="logo-mark">&#9670;</div>
            <h2>Ask anything</h2>
            <p>Search across your indexed tutorials and videos.</p>
          </div>
        )}

        {messages.map(msg => (
          msg.role === 'user' ? (
            <div key={msg.id} className="msg user">
              <div className="bubble user-bubble">{msg.content}</div>
            </div>
          ) : (
            <div key={msg.id} className="msg assistant">
              <div className="bubble assistant-bubble">
                {msg.content ? (
                  <span dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }} />
                ) : msg.streaming ? (
                  <span className="cursor" />
                ) : null}
              </div>

              {msg.sources && msg.sources.length > 0 && (
                <button
                  className="sources-pill"
                  onClick={() => { setActiveSources(msg.sources!); setSourcePanelOpen(true) }}
                >
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                    <path d="M1 6h10M6 1l5 5-5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  {msg.sources.length} source{msg.sources.length !== 1 ? 's' : ''}
                </button>
              )}
            </div>
          )
        ))}

        {statusMsg && (
          <div className="status-row">
            <span className="status-dot" />
            {statusMsg}
          </div>
        )}
      </div>

      <div className="input-bar">
        <div className={`input-wrap ${sending ? 'disabled' : ''}`}>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask about your tutorials..."
            rows={1}
            disabled={sending}
          />
          <div className="input-actions">
            <label className="multihop-toggle">
              <input
                type="checkbox"
                checked={multiHop}
                onChange={e => setMultiHop(e.target.checked)}
              />
              <span className="toggle-track" />
              <span className="toggle-label">Multi-hop</span>
            </label>
            <button
              className="send-btn"
              onClick={send}
              disabled={sending || !input.trim()}
              aria-label="Send"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M14 8L2 2l2.5 6L2 14l12-6z" fill="currentColor"/>
              </svg>
            </button>
          </div>
        </div>
      </div>

      <SourcePanel
        open={sourcePanelOpen}
        sources={activeSources}
        onClose={() => setSourcePanelOpen(false)}
      />
    </div>
  )
}
