# Lore - TODO

**Target:** macOS, Apple Silicon (M1+), 8GB+ RAM
**MCP-native** — no frontend, agents are the UI

## Immediate (before next feature work)
- [ ] Test suite — proper pytest tests, not ad-hoc scripts. Cover: search, entity index, enrichment, ingest, MCP tools

## Cross-Source Connections (Codex-reviewed design)
- [x] **KeywordTagGraph** — sibling to EntityGraph, NPMI on keywords+concept_tags, namespaced nodes (kw:X, tag:Y), Louvain communities, persisted to keyword_graph.json
- [x] **Jaccard similarity index** — precomputed chunk-to-chunk on keyword+tag sets, JSON file
- [x] **Fused find_related** — weighted scoring: entity_overlap + jaccard + keyword/tag overlap

## Wiki Layer (Karpathy LLM Wiki pattern)

### MVP — done
- [x] **Infrastructure** — ~/.lore/wiki/ directory layout, page schema with YAML frontmatter, pages.json manifest, backlinks.json
- [x] **Source pages** — thin wiki wrapper around existing book_summary.json, no new LLM calls
- [x] **Entity pages** — one per canonical EntityIndex cluster. Haiku distills evidence, sonnet synthesizes cross-source.
- [x] **Concept pages** — one per recurring concept_tag/keyword cluster. Haiku distills, sonnet synthesizes.
- [x] **Evidence selection** — deterministic candidate selectors using CrossSourceIndex postings, EntityGraph neighbors, KeywordTagGraph communities
- [x] **Claim-level provenance** — each claim stores chunk_ids + source count. Corroboration: low/moderate/high/mixed.
- [x] **Cross-references** — page→page, page→source, page→chunks. Backlinks manifest.
- [x] **Wiki search** — separate LanceDB wiki_pages table, page fragments indexed, hybrid vector + FTS with RRF fusion
- [x] **Dirty-page invalidation** — ingest/delete marks affected pages stale via collection-aware propagation
- [x] **MCP tools** — wiki_search, wiki_get_page, wiki_generate_page, wiki_related, wiki_claims, wiki_queue (20 tools total)
- [x] **Hybrid triggers** — auto-generate source page + top 10 concepts/entities after ingest, rest on-demand

### Phase 2
- [x] **Comparison pages** — on-demand synthesis comparing 2-4 sources on a topic. 4-stage pipeline: evidence selection (search + postings per collection) → per-source distillation (haiku) → cross-source comparison (sonnet) → page synthesis (sonnet). Collection filter added to search().
- [ ] **Lint/audit tool** — contradiction detection, orphan pages, stale claims, weakly supported claims, missing backlinks, source-gap suggestions. Biggest failure mode protection. (Karpathy emphasizes this explicitly.)
- [ ] **Search ranking heuristics** — query intent detection to route wiki vs chunks. "compare/differ/vs" → boost comparison pages, "what is/explain" → boost concept pages, "exact quote/page number" → suppress wiki, boost raw chunks. Simple keyword matching covers 80%.
- [ ] **Claims-only mode on wiki_get_page** — `include_content=False` to skip prose and return just structured claims. Concept/entity pages are most useful to agents as claim indexes, not prose. Cuts response tokens in half.
- [ ] **Comparison preview** — lightweight `wiki_compare_preview(topic, collections)` that runs evidence selection only (no LLM) and returns chunk overlap stats. Lets agents decide if a full comparison is worth the cost before committing 4 LLM calls.
- [ ] **Ingest compounding** — when a source is ingested, proactively update existing entity/concept pages that reference overlapping entities/tags, not just mark stale. Karpathy: "a single source typically touches 10-15 wiki pages."
- [ ] **Q&A → wiki filing** — mechanism for agent answers to be filed back as new wiki pages so explorations compound the knowledge base. Core Karpathy compounding pattern.
- [ ] **Typed relationships** — claim/page-level semantic relations: supports, contradicts, supersedes, extends. Current backlinks only say "connected", not how. Step toward LLM Wiki v2.
- [ ] **Activity log** — append-only log.md or SQLite equivalent recording all ingests, wiki generations, queries. Karpathy's log.md pattern. Overlaps with session event log in feedback loop.

### Phase 3 — scale
- [ ] **Recursive wiki generation** — autonomous "discover and write all missing pages"
- [ ] **Claim contradiction resolution** — multi-source disagreement detection and surfacing
- [ ] **Page hierarchy + index surfaces** — parent-child taxonomies, community-based auto-grouping, navigable index.md equivalent. Essential at scale (thousands of pages).
- [ ] **Demand-driven generation** — part of feedback loop: session log signals trigger page creation for popular topics
- [ ] **Wiki-maintenance schema** — single governance document (Layer 3) encoding ingest norms, update rules, consistency policies, page-writing conventions. Currently scattered across design docs.

## Agent Feedback Loop (replaces rate_result)
Design: session-level implicit telemetry + explicit `cite` signal. Log passively, materialize scores lazily, boost ranking gently. Demand-driven wiki generation consumes these signals.
- [ ] **Session event log** — append-only SQLite: every search, get_context, wiki_get_page, search_deep logged with session_id + timestamp
- [ ] **`cite` tool** — replaces rate_result. Agent passes list of chunk_ids it used in its answer. Low friction, positive-only signal.
- [ ] **`mark_gap` tool** — agent signals "I needed X and it doesn't exist." Feeds ingest priority queue + wiki page candidates.
- [ ] **Implicit signal extraction** — open_rate (opened/shown), abandon_rate (opened then re-queried), search_deep fallback rate per query pattern
- [ ] **Materialized chunk scores** — background/lazy job computes per-chunk priors: citation_count, open_rate, co_citation pairs
- [ ] **Ranking integration** — small RRF boost from historical citation count as tiebreaker
- [ ] **Wiki auto-suggest** — topics with 3+ search_deep calls across sessions become wiki page candidates
- [ ] **Enrichment feedback** — flag chunks needing re-chunking when many opens cluster on adjacent chunks
- [ ] **Critical mass detection** — auto-enable ranking boost once enough citation data accumulates, avoid polluting rankings with noisy small-sample data

### Session Intelligence (future)
- [ ] "Related" section in search results — Rocchio + MMR recommendations, labeled with WHY
- [ ] Implicit feedback via Rocchio + MMR — centroid of fetched-chunk embeddings, MMR for diversity
- [ ] Co-citation model — chunks frequently cited together get evidence selection priority
- [ ] Upgrade to Thompson Sampling — stochastic exploration for uncertain chunks

## Enrichment
- [ ] Always chunked output — consistent 5000 tok passes
- [ ] Conversation threading across stages — run stages 2-4 as a single LLM conversation so each stage builds on prior context, not independent calls
- [ ] Multilingual NER model (spaCy en_core_web_sm is English-only)

## Model Routing
- [ ] Rate limit fallback: 429 tries next model in chain
- [ ] Local model override via Ollama
- [ ] Token/cost tracking per stage

## Source Structure
- [ ] Auto-generated section labels for videos without chapters (LLM topic shift detection)

## Extractors
- [ ] PDF: fix code block fragmentation on blank lines within code

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
- [x] Entity co-occurrence graph — NPMI weighted edges, Louvain community detection, entity_graph MCP tool, persisted to entity_graph.json
- [x] Built-in dedup — search and get_context never return duplicate content, TTL-aware expiry
- [x] Compact-only MCP search — agents get metadata only, expand via get_context
- [x] Safe schema migration — never drop table on column add failure, just warn and continue
- [x] Ingestion resume — log_ingest_start/log_ingest_status to ingestion_log table, get_resumable_ingests()
- [x] README rewrite — portfolio-oriented, architecture diagram, enrichment v2, entity index, agent experience
- [x] MCP server: 13 tools (added entity_graph tool)
- [x] Model lifecycle manager — JIT loading with TTL eviction (5min sweep), refcounted leasing, ~5GB freed after ingest
- [x] KeyBERT uses shared EmbeddingGemma — eliminated separate all-MiniLM-L6-v2 model (~300MB saved), 2.4x faster extraction
- [x] Query expansion — LLM rewrites queries into variant phrasings, RRF fusion, expand_query param on MCP search tool, in-memory cache, multi-hop skips expansion
- [x] Claude CLI provider — zero-config LLM via host subscription (spawns claude subprocess, inherits OAuth token), auto-detected by registry, sequential execution, model selection (haiku/sonnet/opus)
- [x] Zero-config enrichment — ClaudeProvider replaces MCP sampling plan. Fallback: Claude CLI -> CustomProvider -> skip LLM
- [x] Per-stage model routing — haiku for stage 2 (chunk titles), sonnet for stages 3-4 (summaries). Configurable via enrichment.model_stage{2,3,4}
- [x] Wiki Phase 3 — MCP tools (wiki_search, wiki_get_page, wiki_generate_page, wiki_related, wiki_claims, wiki_queue), LanceDB search index, staleness propagation, codex-reviewed
