<script module lang="ts">
  function renderMarkdown(text: string): string {
    return text
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/```[\w]*\n([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n/g, '<br>')
  }
</script>

<script lang="ts">
  import { tick } from 'svelte'
  import { activeSessionId, sessions } from '../lib/stores'
  import { streamChat, getSessions, getSession } from '../lib/api'
  import type { Message, SearchResult } from '../lib/types'
  import SourcePanel from './SourcePanel.svelte'

  let messages = $state<Message[]>([])
  let input = $state('')
  let sending = $state(false)
  let statusMsg = $state('')
  let multiHop = $state(false)
  let scrollEl: HTMLElement
  let sourcePanelOpen = $state(false)
  let activeSources = $state<SearchResult[]>([])

  $effect(() => {
    if ($activeSessionId) loadSession($activeSessionId)
    else messages = []
  })
  $effect(() => { messages; tick().then(scrollToBottom) })

  async function loadSession(id: string) {
    const data = await getSession(id)
    messages = (data.messages ?? [])
      .filter((m: any) => m.role !== 'system')
      .map((m: any) => ({
        id: m.id,
        role: m.role as 'user' | 'assistant',
        content: m.content,
        sources: m.sources ?? [],
      }))
    await tick()
    scrollToBottom()
  }

  function scrollToBottom() {
    if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight
  }

  async function send() {
    const text = input.trim()
    if (!text || sending) return

    input = ''
    sending = true
    statusMsg = ''

    const assistantId = crypto.randomUUID()
    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: text }
    const assistantMsg: Message = { id: assistantId, role: 'assistant', content: '', streaming: true }
    messages = [...messages, userMsg, assistantMsg]

    const history = messages
      .filter(m => !m.streaming)
      .slice(0, -1)
      .map(m => ({ role: m.role, content: m.content }))

    // Helper: update the assistant message immutably so Svelte 5 sees the change
    function updateAssistant(patch: Partial<Message>) {
      messages = messages.map(m => m.id === assistantId ? { ...m, ...patch } : m)
    }

    await streamChat(
      [...history, { role: 'user', content: text }],
      $activeSessionId,
      { multiHop },
      {
        onSource: (incoming) => {
          const cur = messages.find(m => m.id === assistantId)
          updateAssistant({ sources: [...(cur?.sources ?? []), ...incoming] })
        },
        onToken: (token) => {
          const cur = messages.find(m => m.id === assistantId)
          updateAssistant({ content: (cur?.content ?? '') + token })
        },
        onStatus: (msg) => { statusMsg = msg },
        onSession: async (id) => {
          activeSessionId.set(id)
          const list = await getSessions()
          sessions.set(list)
        },
        onDone: () => {
          updateAssistant({ streaming: false })
          statusMsg = ''
          sending = false
        },
        onError: (err) => {
          updateAssistant({ content: `Error: ${err}`, streaming: false })

          sending = false
          statusMsg = ''
        },
      }
    )
  }

  function openSources(sources: SearchResult[]) {
    activeSources = sources
    sourcePanelOpen = true
  }

  function handleKey(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }
</script>

<div class="chat-wrap">
  <div class="messages" bind:this={scrollEl}>
    {#if messages.length === 0}
      <div class="empty-state">
        <div class="logo-mark">◈</div>
        <h2>Ask anything</h2>
        <p>Search across your indexed tutorials and videos.</p>
      </div>
    {/if}

    {#each messages as msg (msg.id)}
      {#if msg.role === 'user'}
        <div class="msg user">
          <div class="bubble user-bubble">{msg.content}</div>
        </div>
      {:else}
        <div class="msg assistant">
          <div class="bubble assistant-bubble">
            {#if msg.content}
              {@html renderMarkdown(msg.content)}
            {:else if msg.streaming}
              <span class="cursor"></span>
            {/if}
          </div>

          {#if msg.sources && msg.sources.length > 0}
            <button class="sources-pill" onclick={() => openSources(msg.sources!)}>
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M1 6h10M6 1l5 5-5 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
              {msg.sources.length} source{msg.sources.length !== 1 ? 's' : ''}
            </button>
          {/if}
        </div>
      {/if}
    {/each}

    {#if statusMsg}
      <div class="status-row">
        <span class="status-dot"></span>
        {statusMsg}
      </div>
    {/if}
  </div>

  <div class="input-bar">
    <div class="input-wrap" class:disabled={sending}>
      <textarea
        bind:value={input}
        onkeydown={handleKey}
        placeholder="Ask about your tutorials…"
        rows="1"
        disabled={sending}
      ></textarea>
      <div class="input-actions">
        <label class="multihop-toggle" title="Multi-hop: decompose complex queries">
          <input type="checkbox" bind:checked={multiHop} />
          <span class="toggle-label">Multi-hop</span>
        </label>
        <button class="send-btn" onclick={send} disabled={sending || !input.trim()} aria-label="Send">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M14 8L2 2l2.5 6L2 14l12-6z" fill="currentColor"/>
          </svg>
        </button>
      </div>
    </div>
  </div>
</div>

<SourcePanel bind:open={sourcePanelOpen} sources={activeSources} />

<style>
  .chat-wrap {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: #0d0d11;
  }

  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px 32px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    scroll-behavior: smooth;
  }

  .empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    color: #555;
    padding-top: 80px;
  }

  .logo-mark { font-size: 36px; color: #7c6ef7; margin-bottom: 8px; }
  .empty-state h2 { font-size: 18px; color: #888; font-weight: 500; margin: 0; }
  .empty-state p { font-size: 13px; color: #555; margin: 0; }

  .msg {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-width: 720px;
  }

  .msg.user { align-self: flex-end; align-items: flex-end; }
  .msg.assistant { align-self: flex-start; align-items: flex-start; }

  .bubble {
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 14px;
    line-height: 1.6;
    word-break: break-word;
  }

  .user-bubble {
    background: #7c6ef720;
    border: 1px solid #7c6ef740;
    color: #e0e0e8;
    border-bottom-right-radius: 4px;
  }

  .assistant-bubble {
    background: #16161c;
    border: 1px solid #2a2a32;
    color: #d0d0da;
    border-bottom-left-radius: 4px;
  }

  .assistant-bubble :global(pre) {
    background: #0d0d11;
    border: 1px solid #2a2a32;
    border-radius: 6px;
    padding: 10px 12px;
    overflow-x: auto;
    margin: 8px 0;
    font-size: 12px;
  }

  .assistant-bubble :global(code) {
    background: #0d0d11;
    border-radius: 3px;
    padding: 1px 5px;
    font-size: 12px;
    color: #b39df7;
  }

  .assistant-bubble :global(pre code) { background: none; padding: 0; }

  .cursor {
    display: inline-block;
    width: 8px;
    height: 14px;
    background: #7c6ef7;
    border-radius: 1px;
    animation: blink 1s step-end infinite;
    vertical-align: text-bottom;
  }
  @keyframes blink { 50% { opacity: 0; } }

  .sources-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    color: #7c6ef7;
    background: #7c6ef710;
    border: 1px solid #7c6ef730;
    border-radius: 20px;
    padding: 3px 10px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }
  .sources-pill:hover { background: #7c6ef720; border-color: #7c6ef750; }

  .status-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: #666;
    padding: 4px 0;
  }

  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #7c6ef7;
    flex-shrink: 0;
    animation: pulse 1.2s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

  .input-bar { padding: 16px 32px 24px; flex-shrink: 0; }

  .input-wrap {
    background: #16161c;
    border: 1px solid #2a2a32;
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    transition: border-color 0.15s;
  }
  .input-wrap:focus-within { border-color: #7c6ef760; }
  .input-wrap.disabled { opacity: 0.6; }

  textarea {
    background: none;
    border: none;
    outline: none;
    color: #e0e0e8;
    font-size: 14px;
    font-family: inherit;
    padding: 12px 14px 6px;
    resize: none;
    min-height: 44px;
    max-height: 160px;
    overflow-y: auto;
    line-height: 1.5;
  }
  textarea::placeholder { color: #444; }

  .input-actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 10px 8px;
  }

  .multihop-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    user-select: none;
  }
  .multihop-toggle input { display: none; }

  .toggle-label { font-size: 11px; color: #555; transition: color 0.15s; }
  .multihop-toggle:has(input:checked) .toggle-label { color: #7c6ef7; }

  .multihop-toggle::before {
    content: '';
    display: block;
    width: 24px;
    height: 13px;
    border-radius: 7px;
    background: #2a2a32;
    transition: background 0.15s;
    flex-shrink: 0;
  }
  .multihop-toggle:has(input:checked)::before { background: #7c6ef7; }

  .send-btn {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: #7c6ef7;
    border: none;
    color: white;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: opacity 0.15s, transform 0.1s;
    flex-shrink: 0;
  }
  .send-btn:hover:not(:disabled) { opacity: 0.85; }
  .send-btn:active:not(:disabled) { transform: scale(0.94); }
  .send-btn:disabled { opacity: 0.3; cursor: not-allowed; }
</style>
