<script lang="ts">
  import type { SearchResult } from '../lib/types'

  export let sources: SearchResult[] = []
  export let open = false

  function close() { open = false }

  function formatTs(sec: number) {
    const m = Math.floor(sec / 60)
    const s = sec % 60
    return `${m}:${s.toString().padStart(2, '0')}`
  }
</script>

<!-- Backdrop -->
{#if open}
  <button class="backdrop" on:click={close} aria-label="Close sources"></button>
{/if}

<!-- Panel -->
<aside class="panel" class:open>
  <div class="panel-header">
    <span class="panel-title">Sources</span>
    <button class="close-btn" on:click={close}>✕</button>
  </div>

  <div class="panel-body">
    {#each sources as src, i}
      <div class="source-card">
        <div class="source-meta">
          <span class="source-index">{i + 1}</span>
          <div class="source-info">
            <span class="source-collection">{src.collection_display}</span>
            <span class="source-episode">{src.episode_title}</span>
          </div>
          {#if src.url}
            <a
              class="source-timestamp"
              href={src.url}
              target="_blank"
              rel="noopener noreferrer"
            >
              {src.timestamp}
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                <path d="M1 9L9 1M9 1H3M9 1V7" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
              </svg>
            </a>
          {:else}
            <span class="source-timestamp plain">{src.timestamp}</span>
          {/if}
        </div>
        <p class="source-text">{src.text}</p>
      </div>
    {/each}

    {#if sources.length === 0}
      <p class="empty">No sources for this message.</p>
    {/if}
  </div>
</aside>

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    background: transparent;
    z-index: 49;
    border: none;
    cursor: default;
  }

  .panel {
    position: fixed;
    top: 0;
    right: 0;
    width: 380px;
    height: 100vh;
    background: #1a1a1f;
    border-left: 1px solid #2a2a32;
    z-index: 50;
    display: flex;
    flex-direction: column;
    transform: translateX(100%);
    transition: transform 0.22s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .panel.open { transform: translateX(0); }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px;
    border-bottom: 1px solid #2a2a32;
    flex-shrink: 0;
  }

  .panel-title {
    font-size: 13px;
    font-weight: 600;
    color: #e0e0e8;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  .close-btn {
    background: none;
    border: none;
    color: #666;
    cursor: pointer;
    font-size: 14px;
    padding: 4px 6px;
    border-radius: 4px;
    transition: color 0.15s, background 0.15s;
  }
  .close-btn:hover { color: #e0e0e8; background: #2a2a32; }

  .panel-body {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .source-card {
    background: #111116;
    border: 1px solid #2a2a32;
    border-radius: 8px;
    padding: 12px;
  }

  .source-meta {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 8px;
  }

  .source-index {
    font-size: 11px;
    font-weight: 700;
    color: #7c6ef7;
    background: #7c6ef715;
    border-radius: 4px;
    padding: 2px 6px;
    flex-shrink: 0;
    margin-top: 1px;
  }

  .source-info {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .source-collection {
    font-size: 12px;
    font-weight: 600;
    color: #e0e0e8;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .source-episode {
    font-size: 11px;
    color: #888;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .source-timestamp {
    font-size: 11px;
    font-weight: 600;
    color: #7c6ef7;
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 3px;
    flex-shrink: 0;
    padding: 2px 6px;
    border-radius: 4px;
    background: #7c6ef715;
    transition: background 0.15s;
  }
  .source-timestamp:hover { background: #7c6ef730; }
  .source-timestamp.plain { color: #888; background: #2a2a32; }

  .source-text {
    font-size: 12px;
    color: #aaa;
    line-height: 1.55;
    margin: 0;
  }

  .empty {
    color: #555;
    font-size: 13px;
    text-align: center;
    padding: 40px 0;
  }
</style>
