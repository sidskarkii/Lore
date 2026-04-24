# Lore - TODO

**Target:** macOS, Apple Silicon (M1+), 8GB+ RAM
**MCP-native** — no frontend, agents are the UI

## Immediate (before next feature work)
- [ ] Full re-ingest — store was wiped by schema migration. Re-ingest all 16 books + 4 diverse sources with v2 enrichment prompts
- [ ] Safe schema migration — never drop table on column add failure, just warn and continue
- [ ] Test suite — proper pytest tests, not ad-hoc scripts. Cover: search, entity index, enrichment, ingest, MCP tools
- [ ] README update — reflect MCP-native architecture, 12 tools, setup instructions

## Cross-Source Connections
- [ ] Entity co-occurrence graph weighted by NPMI + Louvain community detection (NetworkX). Finds non-obvious cross-source bridges
- [ ] BERTopic topic clustering on existing embeddings (UMAP+HDBSCAN) with hierarchical mode
- [ ] Bipartite graph projection — chunks through shared entities/keywords/tags
- [ ] Jaccard similarity on keyword/tag sets between chunks — index-time relationship discovery
- [ ] TF-IDF pairwise similarity as RRF signal — ensemble of metrics
- [ ] PMI/NPMI on KeyBERT keywords across documents
- [ ] Stage 5: Cross-source connection tagging — classical ML finds candidates, LLM reviews and adds bridging tags

## Wiki Layer (Karpathy LLM Wiki pattern)
- [ ] **Source summary pages** — one per source, auto-generated after ingestion
- [ ] **Concept pages** — one per major topic, synthesized across all sources
- [ ] **Entity pages** — one per person/org/tool, consolidated from fuzzy-merged NER
- [ ] **Synthesis pages** — cross-cutting comparisons generated on demand
- [ ] **Cross-references** — every wiki page links to related pages
- [ ] **Confidence tracking** — claims tagged high/medium/low by corroborating source count
- [ ] **Lint/audit tool** — periodic health check for orphan pages, contradictions, gaps
- [ ] **Wiki search** — search raw chunks OR wiki pages
- [ ] **Incremental wiki updates** — new source triggers updates to existing pages

## Session Intelligence
- [ ] "Related" section in search results — Rocchio + MMR recommendations, labeled with WHY
- [ ] Implicit feedback via Rocchio + MMR — centroid of fetched-chunk embeddings, MMR for diversity
- [ ] Future: chunk co-occurrence patterns from session logs
- [ ] Weight long sessions higher for learning
- [ ] Critical mass detection — auto-enable RL pipeline at threshold
- [ ] Upgrade to Thompson Sampling — stochastic exploration for uncertain chunks
- [ ] Rating persistence across sessions (SQLite survives restarts)
- [ ] Self-improving pipeline — co-occurrence model, sequence patterns, query-chunk affinity

## Retrieval UX
- [ ] Built-in dedup — search and get_context never return duplicate content
- [ ] Search returns all compact by default; agent selectively expands individual results

## Search
- [ ] Incremental FTS index updates instead of full rebuild on every ingest
- [ ] Query expansion via LLM or synonym embeddings

## Enrichment
- [ ] Always chunked output — consistent 5000 tok passes
- [ ] Same session/conversation thread across progressive passes
- [ ] Reuse EmbeddingGemma for KeyBERT instead of loading separate model
- [ ] Multilingual NER model (spaCy en_core_web_sm is English-only)

## MCP Sampling Integration
- [ ] MCP sampling as DEFAULT for all enrichment stages — zero config, no API key
- [ ] External provider as optional upgrade, not requirement
- [ ] Fallback chain: sampling → configured provider → skip LLM (keywords/entities only)

## Model Routing
- [ ] Task-based routing config per stage
- [ ] Small model for stage 2 (e.g. gemma-3-4b)
- [ ] Big model for stages 3-5 synthesis
- [ ] Rate limit fallback: 429 tries next model in chain
- [ ] Local model override via Ollama
- [ ] Token/cost tracking per stage

## Source Structure
- [ ] Auto-generated section labels for videos without chapters (LLM topic shift detection)

## Extractors
- [ ] PDF: fix code block fragmentation on blank lines within code

## Ingestion
- [ ] Ingestion resume: track completed episodes so crashed ingests can resume

## Provider & Configuration
- [ ] Discoverable provider setup — `configure` MCP tool or first-run wizard
- [ ] Support all OpenAI-compatible APIs (docs + validation)

## Code Intelligence
- [ ] tree-sitter-language-pack for multi-language AST parsing
- [ ] Contextual chunk headers (class name + file path prefix)
- [ ] Symbol table extraction (names + signatures as searchable entities)
- [ ] Collapsed class summaries (method bodies -> { ... })
- [ ] Reference graph + PageRank (Aider approach)
- [ ] Code-to-docs cross-referencing

## Packaging & Distribution
- [ ] pip install lore-kb — PyPI package
- [ ] Usage docs + MCP config examples for Claude Code, Cursor, etc.

## Later
- [ ] **Auto-ingest on WebFetch** — domain whitelist, passive KB building during research
- [ ] **Multimodal document parsing** — Docling/MinerU for images/tables/equations
- [ ] **Local enrichment model** — Gemma 4 E4B via Ollama for fully offline pipeline

## Done
- [x] Multi-stage enrichment pipeline (stages 1-4: classical ML → chunk titles → section summaries → book summary)
- [x] Enrichment pipeline v2 — rolling key dictionary (MDKeyChunker-style), 10 fields per chunk (questions, self_contained, confidence, why_important), concept ledger in Stage 3, concept aggregation in Stage 4
- [x] System prompt rework — dynamic state, numbered retrieval loop with token costs, anti-patterns
- [x] Moderation fallback (403 → nemotron-120b)
- [x] Global data dir (~/.lore/) with LORE_DATA_DIR override
- [x] Source-segregated archive (meta.json, extracted.md, chunks.json, section_summaries.json, book_summary.json)
- [x] Domain-specific chunk IDs (EPUB/PDF/video/code/web formats)
- [x] PDF chapter pattern fallback + page number mapping
- [x] PDF heading validation — rejects numbers, URLs, code, long sentences. Invalid headings merge into previous section
- [x] Async non-blocking ingestion with sequential queue + ingest_status
- [x] Interaction logging to SQLite (search/fetch/rate) + chunk_ratings table
- [x] Enrichment cache, retry queue, rate limiting, singleton models
- [x] Robust JSON extraction v2: backtick fences, trailing commas, single-quoted JSON, prose-wrapped output, longest-match
- [x] EPUB spine-based extraction (EPUB3 compatibility)
- [x] Stdio as default MCP transport (auto-starts with harness)
- [x] .env file support + dotenv loading + collection-level dedup
- [x] Progressive disclosure: compact search results with get_context for full fetch
- [x] Token count estimates + reranker scores in MCP results
- [x] MCP server: 12 tools (intro, search, search_deep, get_context, get_toc, find_related, entity_index, reset_session, ingest, ingest_status, rate_result, delete_collection)
- [x] Tool consolidation (14→12): removed health + list_collections, absorbed into intro
- [x] Fuzzy entity merging via rapidfuzz Jaro-Winkler — structural pattern filter, type correction gazetteer, adaptive merge thresholds, type-aware merging
- [x] Entity-enhanced search — query entities expanded through index, chunk entities resolved to canonical forms
- [x] find_related + entity_index MCP tools for cross-source entity discovery
- [x] intro tool (Layer 2 AX) — collection summaries, themes, tags, health, usage stats, cross-source entities, workflows
- [x] TTL re-eligibility (30 min default) + reset_session tool
- [x] Wilson Score chunk ratings + importance boost + session-aware search
- [x] Critical bug fixes: atomic archive writes, collection dedup race, add_chunks chunk loss, section metadata smear, stage 3 fallback recovery
- [x] Sherpa-onnx STT — replaced faster-whisper, whisper-medium.en default, auto-download from HuggingFace, 30s window segmentation
- [x] yt-dlp metadata — chapters mapped to section_heading, tags merged into keywords, channel/description/upload_date stored
- [x] FlashRank cache moved from /tmp to ~/.lore/models/ (persists across reboots)
- [x] Store schema: channel, upload_date, description columns
- [x] Setup script (scripts/setup.sh) — one-command install with verification
- [x] Codex test suites + collaborative review workflow
- [x] Batch ingest script + 16 books + 4 diverse content types ingested
