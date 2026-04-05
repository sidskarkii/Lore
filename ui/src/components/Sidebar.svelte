<script lang="ts">
  import { onMount } from 'svelte'
  import { sessions, activeSessionId, sidebarView, collections, providers, activeProvider } from '../lib/stores'
  import { getSessions, deleteSession, getCollections, getProviders } from '../lib/api'
  import IngestPanel from './IngestPanel.svelte'

  onMount(async () => {
    const [s, c, p] = await Promise.all([getSessions(), getCollections(), getProviders()])
    sessions.set(s)
    collections.set(c)
    providers.set(p.providers)
    activeProvider.set(p.active)
  })

  function newChat() {
    activeSessionId.set(null)
    sidebarView.set('chat')
  }

  async function removeSession(id: string, e: MouseEvent) {
    e.stopPropagation()
    await deleteSession(id)
    sessions.update(s => s.filter(x => x.id !== id))
    if ($activeSessionId === id) activeSessionId.set(null)
  }

  function formatDate(iso: string) {
    const d = new Date(iso)
    const now = new Date()
    const diff = now.getTime() - d.getTime()
    if (diff < 86400000) return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    if (diff < 604800000) return d.toLocaleDateString([], { weekday: 'short' })
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
  }

  $: activeProviderInfo = $providers.find(p => p.name === $activeProvider)
</script>

<aside class="sidebar">
  <!-- Logo -->
  <div class="brand">
    <span class="brand-icon">◈</span>
    <span class="brand-name">Lore</span>
  </div>

  <!-- Nav -->
  <div class="nav">
    <button class="nav-btn" class:active={$sidebarView === 'chat'} on:click={() => sidebarView.set('chat')}>
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <path d="M1 2a1 1 0 011-1h10a1 1 0 011 1v7a1 1 0 01-1 1H8l-3 3V10H2a1 1 0 01-1-1V2z" stroke="currentColor" stroke-width="1.4"/>
      </svg>
      Chat
    </button>
    <button class="nav-btn" class:active={$sidebarView === 'ingest'} on:click={() => sidebarView.set('ingest')}>
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <path d="M7 1v8M4 6l3 3 3-3M2 11h10" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      Add content
    </button>
    <button class="nav-btn" class:active={$sidebarView === 'library'} on:click={() => sidebarView.set('library')}>
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <rect x="1" y="1" width="5" height="5" rx="1" stroke="currentColor" stroke-width="1.4"/>
        <rect x="8" y="1" width="5" height="5" rx="1" stroke="currentColor" stroke-width="1.4"/>
        <rect x="1" y="8" width="5" height="5" rx="1" stroke="currentColor" stroke-width="1.4"/>
        <rect x="8" y="8" width="5" height="5" rx="1" stroke="currentColor" stroke-width="1.4"/>
      </svg>
      Library
    </button>
  </div>

  <!-- Content area -->
  <div class="content">
    {#if $sidebarView === 'chat'}
      <button class="new-chat-btn" on:click={newChat}>
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
          <path d="M6 1v10M1 6h10" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
        </svg>
        New chat
      </button>

      {#if $sessions.length > 0}
        <div class="section-label">Recent</div>
        <div class="session-list">
          {#each $sessions as session (session.id)}
            <div
              class="session-item"
              class:active={$activeSessionId === session.id}
              role="button"
              tabindex="0"
              on:click={() => { activeSessionId.set(session.id); sidebarView.set('chat') }}
              on:keydown={(e) => e.key === 'Enter' && activeSessionId.set(session.id)}
            >
              <span class="session-title">{session.title}</span>
              <div class="session-meta">
                <span class="session-date">{formatDate(session.updated_at)}</span>
                <button class="del-btn" on:click={(e) => removeSession(session.id, e)} title="Delete" aria-label="Delete session">
                  <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                    <path d="M1 1l8 8M9 1L1 9" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>
                  </svg>
                </button>
              </div>
            </div>
          {/each}
        </div>
      {:else}
        <p class="empty-hint">No conversations yet.</p>
      {/if}

    {:else if $sidebarView === 'ingest'}
      <IngestPanel />

    {:else if $sidebarView === 'library'}
      <div class="section-label" style="padding: 12px 16px 6px">Collections</div>
      {#if $collections.length > 0}
        <div class="collection-list">
          {#each $collections as col}
            <div class="collection-item">
              <div class="col-name">{col.collection_display}</div>
              <div class="col-meta">
                <span class="col-tag">{col.topic}</span>
                {#if col.subtopic}<span class="col-tag">{col.subtopic}</span>{/if}
                <span class="col-count">{col.episode_count} ep</span>
              </div>
            </div>
          {/each}
        </div>
      {:else}
        <p class="empty-hint">No content indexed yet.</p>
      {/if}
    {/if}
  </div>

  <!-- Footer: active provider -->
  <div class="footer">
    {#if activeProviderInfo}
      <div class="provider-badge">
        <span class="provider-dot" class:ok={activeProviderInfo.authenticated}></span>
        <span class="provider-name">{activeProviderInfo.display_name}</span>
      </div>
    {:else}
      <div class="provider-badge">
        <span class="provider-dot"></span>
        <span class="provider-name" style="color:#555">No provider</span>
      </div>
    {/if}
  </div>
</aside>

<style>
  .sidebar {
    width: 240px;
    flex-shrink: 0;
    background: #111116;
    border-right: 1px solid #1e1e26;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 18px 16px 14px;
    border-bottom: 1px solid #1e1e26;
    flex-shrink: 0;
  }

  .brand-icon { font-size: 18px; color: #7c6ef7; }
  .brand-name { font-size: 15px; font-weight: 700; color: #e0e0e8; letter-spacing: -0.02em; }

  .nav {
    display: flex;
    flex-direction: column;
    gap: 1px;
    padding: 8px;
    border-bottom: 1px solid #1e1e26;
    flex-shrink: 0;
  }

  .nav-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    background: none;
    border: none;
    color: #666;
    font-size: 12px;
    font-family: inherit;
    padding: 7px 8px;
    border-radius: 6px;
    cursor: pointer;
    text-align: left;
    transition: background 0.12s, color 0.12s;
    width: 100%;
  }
  .nav-btn:hover { background: #1a1a22; color: #bbb; }
  .nav-btn.active { background: #1e1e2a; color: #e0e0e8; }

  .content {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }

  .new-chat-btn {
    display: flex;
    align-items: center;
    gap: 7px;
    margin: 10px 8px 6px;
    background: #1e1e2a;
    border: 1px solid #2a2a38;
    color: #bbb;
    font-size: 12px;
    font-family: inherit;
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.12s, color 0.12s;
  }
  .new-chat-btn:hover { background: #252532; color: #e0e0e8; }

  .section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #444;
    padding: 8px 16px 4px;
  }

  .session-list {
    display: flex;
    flex-direction: column;
    gap: 1px;
    padding: 0 6px;
  }

  .session-item {
    display: flex;
    flex-direction: column;
    gap: 3px;
    border-radius: 6px;
    padding: 7px 10px;
    cursor: pointer;
    transition: background 0.12s;
    width: 100%;
  }
  .session-item:hover { background: #1a1a22; }
  .session-item.active { background: #1e1e2a; }

  .session-title {
    font-size: 12px;
    color: #ccc;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    display: block;
  }
  .session-item.active .session-title { color: #e0e0e8; }

  .session-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .session-date { font-size: 10px; color: #555; }

  .del-btn {
    background: none;
    border: none;
    color: #555;
    padding: 2px 3px;
    cursor: pointer;
    border-radius: 3px;
    display: flex;
    opacity: 0;
    transition: opacity 0.12s, color 0.12s;
  }
  .session-item:hover .del-btn { opacity: 1; }
  .del-btn:hover { color: #e05a5a; }

  .empty-hint {
    font-size: 12px;
    color: #444;
    padding: 16px;
    margin: 0;
  }

  /* Library */
  .collection-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 0 8px;
  }

  .collection-item {
    background: #16161c;
    border: 1px solid #1e1e26;
    border-radius: 8px;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }

  .col-name { font-size: 12px; color: #ddd; font-weight: 500; }

  .col-meta { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }

  .col-tag {
    font-size: 10px;
    background: #7c6ef715;
    color: #7c6ef7;
    border-radius: 3px;
    padding: 1px 5px;
  }

  .col-count { font-size: 10px; color: #555; margin-left: auto; }

  /* Footer */
  .footer {
    padding: 10px 12px;
    border-top: 1px solid #1e1e26;
    flex-shrink: 0;
  }

  .provider-badge {
    display: flex;
    align-items: center;
    gap: 7px;
  }

  .provider-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #444;
    flex-shrink: 0;
  }
  .provider-dot.ok { background: #4caf7d; }

  .provider-name { font-size: 11px; color: #777; }
</style>
