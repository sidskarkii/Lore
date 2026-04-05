<script lang="ts">
  import { ingestFolder, ingestYouTube, getCollections } from '../lib/api'
  import { collections } from '../lib/stores'
  import type { IngestJob } from '../lib/types'

  type Tab = 'folder' | 'youtube'
  let tab: Tab = 'youtube'

  // Shared fields
  let name = ''
  let topic = ''
  let subtopic = ''

  // Folder
  let folderPath = ''

  // YouTube
  let ytUrl = ''

  let jobs: IngestJob[] = []

  async function runIngest() {
    const label = tab === 'folder' ? folderPath : ytUrl
    if (!label || !name) return

    const job: IngestJob = { type: tab, label, status: 'running', message: 'Ingesting…' }
    jobs = [job, ...jobs]

    try {
      const res = tab === 'folder'
        ? await ingestFolder({ path: folderPath, name, topic, subtopic })
        : await ingestYouTube({ url: ytUrl, name, topic, subtopic })

      if (res.success) {
        job.status = 'done'
        job.message = `Done — ${res.chunks} chunks indexed`
        const updated = await getCollections()
        collections.set(updated)
      } else {
        job.status = 'error'
        job.message = res.error ?? 'Unknown error'
      }
    } catch (e: any) {
      job.status = 'error'
      job.message = e.message ?? 'Request failed'
    }

    jobs = jobs
  }

  $: canSubmit = !!name && (tab === 'folder' ? !!folderPath : !!ytUrl)
</script>

<div class="ingest">
  <div class="section-header">Add content</div>

  <div class="tabs">
    <button class="tab" class:active={tab === 'youtube'} on:click={() => tab = 'youtube'}>
      YouTube
    </button>
    <button class="tab" class:active={tab === 'folder'} on:click={() => tab = 'folder'}>
      Folder
    </button>
  </div>

  <form class="form" on:submit|preventDefault={runIngest}>
    {#if tab === 'youtube'}
      <label class="field">
        <span>URL</span>
        <input bind:value={ytUrl} placeholder="https://youtube.com/watch?v=…" type="url" />
      </label>
    {:else}
      <label class="field">
        <span>Folder path</span>
        <input bind:value={folderPath} placeholder="C:\Videos\BlenderTuts" />
      </label>
    {/if}

    <label class="field">
      <span>Collection name</span>
      <input bind:value={name} placeholder="Blender Rigging Series" />
    </label>

    <div class="row">
      <label class="field">
        <span>Topic</span>
        <input bind:value={topic} placeholder="3d" />
      </label>
      <label class="field">
        <span>Subtopic</span>
        <input bind:value={subtopic} placeholder="blender" />
      </label>
    </div>

    <button class="submit-btn" type="submit" disabled={!canSubmit}>
      Start ingestion
    </button>
  </form>

  {#if jobs.length > 0}
    <div class="jobs">
      <div class="section-header" style="margin-top: 8px;">Recent</div>
      {#each jobs as job}
        <div class="job-row" class:done={job.status === 'done'} class:error={job.status === 'error'}>
          <div class="job-indicator" class:running={job.status === 'running'}></div>
          <div class="job-info">
            <span class="job-label">{job.label}</span>
            <span class="job-msg">{job.message}</span>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .ingest {
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .section-header {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #555;
    padding: 4px 4px 8px;
  }

  .tabs {
    display: flex;
    gap: 4px;
    background: #111116;
    border-radius: 8px;
    padding: 3px;
  }

  .tab {
    flex: 1;
    background: none;
    border: none;
    color: #666;
    font-size: 12px;
    padding: 6px 0;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
    font-family: inherit;
  }
  .tab.active { background: #2a2a38; color: #e0e0e8; }
  .tab:hover:not(.active) { color: #aaa; }

  .form {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .field {
    display: flex;
    flex-direction: column;
    gap: 4px;
    flex: 1;
  }

  .field span {
    font-size: 11px;
    color: #666;
    font-weight: 500;
  }

  .field input {
    background: #111116;
    border: 1px solid #2a2a32;
    border-radius: 6px;
    color: #e0e0e8;
    font-size: 12px;
    padding: 7px 10px;
    outline: none;
    font-family: inherit;
    transition: border-color 0.15s;
  }
  .field input:focus { border-color: #7c6ef760; }
  .field input::placeholder { color: #444; }

  .row {
    display: flex;
    gap: 8px;
  }

  .submit-btn {
    background: #7c6ef7;
    border: none;
    color: white;
    font-size: 12px;
    font-weight: 600;
    padding: 9px;
    border-radius: 8px;
    cursor: pointer;
    font-family: inherit;
    transition: opacity 0.15s;
    margin-top: 4px;
  }
  .submit-btn:hover:not(:disabled) { opacity: 0.85; }
  .submit-btn:disabled { opacity: 0.35; cursor: not-allowed; }

  .jobs {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .job-row {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #111116;
    border: 1px solid #2a2a32;
    border-radius: 8px;
    padding: 10px 12px;
  }
  .job-row.done { border-color: #2a3a2a; }
  .job-row.error { border-color: #3a2a2a; }

  .job-indicator {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #666;
    flex-shrink: 0;
  }
  .job-indicator.running {
    background: #7c6ef7;
    animation: pulse 1.2s ease-in-out infinite;
  }
  .job-row.done .job-indicator { background: #4caf7d; }
  .job-row.error .job-indicator { background: #e05a5a; }

  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

  .job-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }

  .job-label {
    font-size: 11px;
    color: #bbb;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .job-msg {
    font-size: 11px;
    color: #666;
  }
</style>
