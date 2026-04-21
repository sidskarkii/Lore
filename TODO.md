# Lore - TODO

**Target:** macOS, Apple Silicon (M1+), 16GB+ RAM

## Context Engineering
- [ ] Entity-boosted search: extract entities from query, match against stored entities, boost chunks with entity overlap as third RRF signal
- [ ] System prompt rework for RAG harness

## Source Structure & Navigation
- [ ] Domain-specific chunk IDs and hierarchy: books use part/chapter/section (e.g. `pt01_ch03_s005`), videos keep episode/timestamp, code uses file/symbol. get_context understands the source's natural structure ("next chapter", "this part") instead of raw index offsets
- [ ] Video chapter extraction: pull YouTube chapters from description metadata (yt-dlp), store as chapter field on video chunks so compact results show section names instead of bare timestamps
- [ ] Auto-generated section labels: LLM enrichment detects topic shifts in videos without chapters, assigns section names during enrichment pass

## Cross-Source Connections
All computed locally from existing data (embeddings, NER entities, keywords) — no LLM calls needed.
- [ ] Fuzzy entity merging via rapidfuzz (Jaro-Winkler) — prerequisite for entity work ("Einstein" / "A. Einstein" / "Albert Einstein" -> same entity)
- [ ] Entity co-occurrence graph weighted by NPMI + Louvain community detection (NetworkX). Finds non-obvious cross-source bridges. See BlueGraph for reference
- [ ] BERTopic topic clustering on existing embeddings (UMAP+HDBSCAN) with hierarchical mode for multi-level granularity (chunk < subtopic < topic < theme). Known M1 stability issues at scale — test early, fall back to UMAP+GMM (RAPTOR approach) if needed
- [ ] Bipartite graph projection — project chunks through shared entities, keywords, or tags for multi-dimensional chunk-chunk connections. Build after entity co-occurrence to evaluate incremental value
- [ ] Jaccard similarity on keyword/tag sets between chunks — index-time relationship discovery (different from BM25 which is query-time matching)
- [ ] TF-IDF pairwise similarity as third RRF signal — ensemble of metrics gives ~40% improvement over single metric (USMB 2025). Different signal from embedding cosine even if overlap exists
- [ ] PMI/NPMI on KeyBERT keywords across documents — finds statistically surprising associations between sources. Needs enrichment running first
- [ ] `find_related(chunk_id)` / `find_connections(collection)` MCP tools to expose connections to agents

## Agent Alignment

### Agent Onboarding (AX — agent experience)
Progressive disclosure for the agent itself: base knowledge automatic, deeper knowledge available, power features opt-in.
- [x] Layer 1: Dynamic MCP instructions — on connect, agent automatically sees: collection count, topic list, total chunks, one-line usage guide. No tool call needed, just a smarter instructions string that reads current state
- [ ] Layer 2: `intro` tool — on-demand deep orientation. Returns structured data: collections with summaries, suggested workflows (compact search -> expand -> get_context), ingestion nudges ("if you find YouTube links or PDFs on this topic, ingest them for deeper research"). Makes the agent want to use Lore proactively
- [ ] Layer 3: `/lore` Claude Code skill — loads full workflow prompt template. Teaches optimal patterns, available tools, tips. Power user opt-in, similar to claude-mem's skill approach

### Retrieval UX
- [x] `get_toc(collection)` tool — returns document structure (parts, chapters, sections) for structural navigation
- [ ] Paginated context expansion — `get_context` returns pages instead of dumping full neighborhood. Agent controls page size via `page_tokens` param (e.g. 50 or 1000 tokens per page) and navigates with `page` param. Each response includes total pages so agent knows how much is left
- [ ] Built-in dedup — search and get_context never return content the agent already has in the current response
- [ ] Search returns all compact by default; agent selectively expands individual results (current compact/expand model but with pagination on expand side)

### Session Intelligence
- [ ] Session-aware search — combo approach: (1) deprioritize seen chunks by 50% score penalty instead of filtering (avoids losing top results after agent compaction), (2) TTL-based re-eligibility (chunks become full-score again after N minutes), (3) `reset_session` param for agent to signal compaction happened
- [ ] "Related" section in search results — Rocchio + MMR recommendations shown separately from main results, clearly labeled with WHY ("similar to chunks you read about X and Y"). Main results stay unbiased, related section adds session-informed suggestions
- [ ] Implicit feedback via Rocchio + MMR — maintain centroid of fetched-chunk embeddings per session, retrieve nearest neighbors, rerank with MMR (lambda ~0.7) to ensure diversity. Prevents echo chamber while keeping relevance. Use ONLY for the "related" section, never bias main results
- [ ] Future: chunk co-occurrence patterns from session logs — learn which chunks get fetched together across sessions as blended signal for recommendations

### Interaction Logging (prerequisite for all learning)
- [ ] Log every agent interaction to SQLite: `(session_id, timestamp, action, query, chunk_ids_shown, chunk_ids_fetched, chunk_ids_ignored, chunk_ids_rated)`. Ignored = shown but not fetched. This is the training data for everything below — log from day one
- [ ] Weight long sessions higher — a session with 100+ interactions is worth more than 5 sessions with 3 each. Track interaction count per session for weighting
- [ ] Critical mass detection — periodically count sessions/interactions. Once threshold hit (e.g. 500+ sessions or equivalent weighted), enable RL/co-occurrence training pipeline automatically

### Long-Term Learning
- [ ] Wilson Score chunk ratings — store (chunk_id, fetches, ignores) per chunk in SQLite. Rank by lower bound of confidence interval so low-data chunks don't dominate. Binary signal (fetched vs shown-but-ignored) maps perfectly. 5 lines of scipy, zero dependencies
- [ ] `rate_result(chunk_id, useful)` tool — explicit feedback as strong signal, counts as multiple implicit fetches/ignores. Stored in same SQLite table
- [ ] Upgrade to Thompson Sampling — same data model (alpha=fetches, beta=ignores) but adds stochastic exploration. Uncertain chunks occasionally surface higher to gather signal. Switch when we want active exploration behavior
- [ ] Rating persistence across sessions — SQLite table survives server restarts, improves ranking over weeks of use
- [ ] Self-improving pipeline — when critical mass hit: train chunk co-occurrence model, session sequence patterns, query-chunk affinity. Retrain periodically as more data accumulates. Log format designed to be consumable by RL frameworks

## Wiki Layer (Karpathy LLM Wiki pattern)
Synthesized knowledge layer on top of raw chunks. Classical ML finds connections, LLM writes the pages.
- [ ] **Source summary pages** — one per ingested source, auto-generated after ingestion. Key points, concepts mentioned, entities, confidence. Stored as searchable chunks alongside raw content
- [ ] **Concept pages** — one per major topic (e.g. "transformer attention", "social proof"). Synthesized across ALL sources that mention the concept. Updated when new sources add info. BERTopic clusters identify which concepts exist, LLM writes the page
- [ ] **Entity pages** — one per person/org/tool. Consolidated from fuzzy-merged NER entities across sources. LLM writes overview, characteristics, related concepts
- [ ] **Synthesis pages** — cross-cutting comparisons generated on demand ("compare how Book A and Book B discuss leadership"). Created when agent asks, persisted for future queries
- [ ] **Cross-references** — every wiki page links to related pages. Built from entity co-occurrence graph + embedding similarity
- [ ] **Confidence tracking** — claims tagged high/medium/low based on number of corroborating sources. Multiple sources saying the same thing = high confidence
- [ ] **Lint/audit tool** — periodic health check: find orphan pages, contradictions between sources, gaps, stale claims. Auto-fix what's possible, report the rest
- [ ] **Wiki search** — agent can search raw chunks OR wiki pages. Wiki pages surface synthesized knowledge, raw chunks surface source material
- [ ] **Incremental wiki updates** — new source ingested triggers updates to existing concept/entity pages, not just creation of new ones

## Smart Model Routing
Route LLM tasks to appropriately-sized models via OpenRouter. Cheap models for simple tasks, big models for complex ones.
- [ ] Task-based routing config in config.yaml: `routing.summarize`, `routing.synthesize`, `routing.reason`, `routing.default`
- [ ] Small model for chunk summaries/titles/tags (e.g. `gemma-3-4b-it:free` — fast, good enough)
- [ ] Big model for concept page synthesis, entity consolidation, contradiction detection (e.g. `gpt-oss-120b:free` or `nemotron-3-super-120b`)
- [ ] Fallback chain: if preferred model is rate-limited, try next in list
- [ ] Token/cost tracking per task type for monitoring which routes are expensive
- [ ] Local model override: when Ollama is available, route simple tasks to local Gemma 4 E4B instead of API

## Search
- [ ] Incremental FTS index updates instead of full rebuild on every ingest
- [ ] Query expansion via LLM or synonym embeddings

## Features
- [ ] Store source documents as cleaned markdown after extraction for UI reader view
- [ ] Settings UI: expose all configurable knobs in the frontend

## Transcription
- [ ] Replace faster-whisper with sherpa-onnx (ONNX Runtime, CoreML on Apple Silicon)
- [ ] Default to lightweight model (Moonshine tiny ~55MB or Whisper tiny ~75MB)

## Code Intelligence
- [ ] tree-sitter-language-pack for multi-language AST parsing
- [ ] Contextual chunk headers (class name + file path prefix)
- [ ] Symbol table extraction (names + signatures as searchable entities)
- [ ] Collapsed class summaries (method bodies -> { ... })
- [ ] Reference graph + PageRank (Aider approach)
- [ ] Code-to-docs cross-referencing — match symbol names from tree-sitter (classes, functions) against mentions in ingested docs. Feeds into entity co-occurrence graph as high-confidence connections

## Extractors
- [ ] PDF: tune heading detection heuristics (some figure captions picked up as headings)
- [ ] PDF: fix code block fragmentation on blank lines within code

## Ingestion
- [ ] Ingestion resume: track completed episodes so crashed ingests can resume

## Enrichment
- [ ] Reuse EmbeddingGemma for KeyBERT instead of loading separate model (~80MB savings)
- [ ] Structured output / JSON mode for LLM enrichment instead of code fence parsing
- [ ] Multilingual NER model (spaCy en_core_web_sm is English-only)

## Later
- [ ] Frontend rework
- [ ] DMG packaging via Tauri sidecar
- [ ] Local enrichment model — Gemma 4 E4B via Ollama (~5GB Q4, ~40-60 tok/s on M1, native JSON output). Runner-up: Qwen 3.5 4B (~2.5GB, faster). Replace OpenRouter enrichment calls with local inference for fully offline pipeline
- [ ] MCP sampling for enrichment (use calling agent's LLM)

## Done
- [x] Progressive disclosure: compact search results (metadata + scores + token_count, no text) with get_context for full fetch
- [x] Token count estimates in search results
- [x] Reranker scores passed through to MCP results
- [x] Configurable host/port from config.yaml
- [x] CORS regex instead of hardcoded origin list
- [x] Chat route refactor: extracted shared helpers, dropped WebSocket endpoint
- [x] Inline imports cleanup
- [x] Removed all truncations in reranker, enrichment, contextual prefixes
- [x] MCP server: 7 tools, streamable HTTP + stdio
- [x] Search expand parameter + get_chunk_by_id
- [x] EPUB extractor: recursive headings + code block preservation
- [x] PDF extractor: font-aware code detection + heading detection, no OCR
- [x] Provider cleanup: simplified to custom/OpenRouter only
- [x] Default model: openrouter/elephant-alpha
- [x] Source-specific location fields (page, section, chapter, lines, chunk_index)
- [x] Document expansion via chunk_index (fixes timestamp=0 bug)
- [x] Ingestion consolidation: shared _ingest_extracted helper
- [x] Chunk deduplication via content hash
- [x] Batch LLM enrichment with tags
- [x] Windows paths removed
- [x] Web extractor markdown output
- [x] get_context fix for document sources
- [x] Full pipeline tested end-to-end
