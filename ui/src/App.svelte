<script lang="ts">
  import { getHealth } from './lib/api'
  import Sidebar from './components/Sidebar.svelte'
  import ChatView from './components/ChatView.svelte'

  type Status = 'loading' | 'ready' | 'error'
  type LogEntry = { text: string; done: boolean }

  let status = $state<Status>('loading')
  let errorMsg = $state('')
  let logs = $state<LogEntry[]>([])

  function addLog(text: string, done = false) {
    logs = [...logs, { text, done }]
  }

  function markLastDone() {
    if (logs.length === 0) return
    logs = logs.map((l, i) => i === logs.length - 1 ? { ...l, done: true } : l)
  }

  async function warmup() {
    status = 'loading'
    errorMsg = ''
    logs = []

    addLog('Connecting to backend…')
    try {
      const h = await getHealth()
      markLastDone()

      if (h.embedding_model) addLog(`Embedding model: ${h.embedding_model}`, true)
      if (h.reranker_model)  addLog(`Reranker: ${h.reranker_model}`, true)
      if (h.active_provider) addLog(`Provider: ${h.active_provider}`, true)
      addLog(`${h.total_chunks} chunks indexed`, true)

      // Brief pause so logs are readable before the app appears
      await new Promise(r => setTimeout(r, 600))
      status = 'ready'
    } catch {
      markLastDone()
      status = 'error'
      errorMsg = 'Cannot reach the Lore backend.\n\nMake sure it\'s running:\n\npython -m lore'
    }
  }

  warmup()
</script>

{#if status === 'loading' || status === 'error'}
  <div class="splash">
    <div class="splash-icon" class:error={status === 'error'}>◈</div>

    {#if status === 'loading'}
      <p class="splash-title">Starting up</p>
      <div class="spinner"></div>
    {:else}
      <p class="splash-title error-text">Backend not available</p>
    {/if}

    {#if logs.length > 0}
      <div class="log-box">
        {#each logs as entry}
          <div class="log-line" class:done={entry.done}>
            <span class="log-dot" class:done={entry.done}></span>
            <span>{entry.text}</span>
          </div>
        {/each}
      </div>
    {/if}

    {#if status === 'error'}
      <pre class="splash-error">{errorMsg}</pre>
      <button class="retry-btn" onclick={warmup}>Retry</button>
    {/if}
  </div>

{:else}
  <div class="app">
    <Sidebar />
    <main class="main">
      <ChatView />
    </main>
  </div>
{/if}

<style>
  :global(*, *::before, *::after) { box-sizing: border-box; margin: 0; padding: 0; }
  :global(body) {
    background: #0d0d11;
    color: #e0e0e8;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
    overflow: hidden;
  }
  :global(::-webkit-scrollbar) { width: 5px; height: 5px; }
  :global(::-webkit-scrollbar-track) { background: transparent; }
  :global(::-webkit-scrollbar-thumb) { background: #2a2a38; border-radius: 3px; }
  :global(::-webkit-scrollbar-thumb:hover) { background: #3a3a50; }

  .app {
    display: flex;
    height: 100vh;
    width: 100vw;
    overflow: hidden;
  }

  .main {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* Splash */
  .splash {
    height: 100vh;
    width: 100vw;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 14px;
    background: #0d0d11;
  }

  .splash-icon {
    font-size: 40px;
    color: #7c6ef7;
    margin-bottom: 2px;
  }
  .splash-icon.error { color: #e05a5a; }

  .splash-title {
    font-size: 15px;
    font-weight: 500;
    color: #ccc;
  }
  .splash-title.error-text { color: #e05a5a; }

  .spinner {
    width: 18px;
    height: 18px;
    border: 2px solid #2a2a38;
    border-top-color: #7c6ef7;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .log-box {
    background: #111116;
    border: 1px solid #1e1e26;
    border-radius: 10px;
    padding: 12px 16px;
    width: 300px;
    display: flex;
    flex-direction: column;
    gap: 7px;
  }

  .log-line {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: #555;
    transition: color 0.2s;
  }
  .log-line.done { color: #aaa; }

  .log-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #333;
    flex-shrink: 0;
    transition: background 0.2s;
  }
  .log-dot.done { background: #4caf7d; }

  .splash-error {
    font-size: 12px;
    color: #666;
    background: #111116;
    border: 1px solid #2a2a32;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
    white-space: pre-wrap;
    font-family: inherit;
    width: 300px;
    line-height: 1.7;
  }

  .retry-btn {
    background: #1e1e2a;
    border: 1px solid #2a2a38;
    color: #bbb;
    font-size: 12px;
    font-family: inherit;
    padding: 8px 20px;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
  }
  .retry-btn:hover { background: #252532; color: #e0e0e8; }
</style>
