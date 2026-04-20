# Lore - TODO

**Target:** macOS, Apple Silicon (M1+), 16GB+ RAM

## Context Engineering
- [ ] Entity-boosted search: extract entities from query, match against stored entities, boost chunks with entity overlap as third RRF signal
- [ ] System prompt rework for RAG harness

## Source Structure & Navigation
- [ ] Domain-specific chunk IDs and hierarchy: books use part/chapter/section (e.g. `pt01_ch03_s005`), videos keep episode/timestamp, code uses file/symbol. get_context understands the source's natural structure ("next chapter", "this part") instead of raw index offsets
- [ ] Video chapter extraction: pull YouTube chapters from description metadata (yt-dlp), store as chapter field on video chunks so compact results show section names instead of bare timestamps
- [ ] Auto-generated section labels: LLM enrichment detects topic shifts in videos without chapters, assigns section names during enrichment pass

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
- [ ] Local small model for offline UI chat (Gemma)
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
