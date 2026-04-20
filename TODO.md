# Lore - TODO

**Target:** macOS, Apple Silicon (M1+), 16GB+ RAM

## Context Engineering
- [ ] Progressive disclosure in search: return compact results first (title, section, collection, chunk_id, token_count), agent fetches full text via get_context. Balance between useful-at-a-glance and not requiring extra tool calls for simple queries
- [ ] Entity-boosted search: extract entities from query, match against stored entities, boost chunks with entity overlap as third RRF signal
- [ ] Token count estimates in search results so agent knows context budget cost before fetching
- [ ] System prompt rework for RAG harness

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

## Extractors
- [ ] Web: switch trafilatura to markdown output (done, needs broader testing)
- [ ] PDF: tune heading detection heuristics (some figure captions picked up as headings)
- [ ] PDF: fix code block fragmentation on blank lines within code

## Ingestion
- [ ] Ingestion resume: track completed episodes so crashed ingests can resume
- [ ] Consolidate remaining old ingest paths fully

## Enrichment
- [ ] Reuse EmbeddingGemma for KeyBERT instead of loading separate model (~80MB savings)
- [ ] Structured output / JSON mode for LLM enrichment instead of code fence parsing
- [ ] Multilingual NER model (spaCy en_core_web_sm is English-only)

## Later
- [ ] Frontend rework
- [ ] DMG packaging via Tauri sidecar
- [ ] Local small model for offline UI chat (Gemma)
- [ ] MCP sampling for enrichment (use calling agent's LLM)

## Done
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
