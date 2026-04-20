# Lore - Rework TODO

**Target platform:** macOS only, Apple Silicon (M1+), 16GB+ RAM

## Architecture
- [ ] Rich search returns: relevance scores, cross-references to related chunks in other collections, entity overlap with query
- [ ] Local small model for UI chat (late stage): bundle small Gemma for offline use, OpenRouter as default
- [ ] Settings UI: expose all configurable knobs (models, search params, chunking, transcription, provider) in the frontend settings page

## Search
- [ ] Incremental FTS index updates instead of full rebuild on every ingest (replace=True gets slow at scale)
- [ ] Query expansion: use LLM or synonym embeddings to expand queries before search ("donut" should match "torus modeling")

## Features
- [ ] Store source documents: save cleaned markdown/txt of full content after extraction so UI can show a reader view
- [ ] System prompt rework: current one is too naive, needs proper context engineering for the RAG harness

## Transcription
- [ ] Replace faster-whisper with sherpa-onnx: reuses existing ONNX Runtime, supports Whisper + Moonshine + Paraformer, CoreML on Apple Silicon
- [ ] Default to a lightweight model (Moonshine tiny ~55MB or Whisper tiny ~75MB) for out-of-box experience

## Extractors
- [ ] Code: add tree-sitter-language-pack (~2.5MB) for proper AST parsing of JS, TS, Go, Rust, Java, etc.
- [ ] Code: contextual chunk headers (class name + file path prefix)
- [ ] Code: symbol table extraction (function/class names + signatures as separate searchable entities)
- [ ] Code: collapsed class summaries (method bodies replaced by { ... })
- [ ] Code: reference graph + PageRank (Aider approach)
- [ ] Web: switch trafilatura output_format from "txt" to "markdown" to preserve headings
- [ ] PDF (optional): consider Docling for complex docs with tables/multi-column layouts

## Ingestion Pipeline
- [ ] Remove hardcoded Windows paths, macOS only
- [ ] Consolidate ingest_documents, ingest_file, ingest_url into one clean path
- [ ] Chunk deduplication: hash chunk text and skip duplicates
- [ ] Ingestion resume: track completed episodes so crashed ingests can resume

## Enrichment
- [ ] Add tags to LLM enrichment output
- [ ] Consider reusing EmbeddingGemma for KeyBERT instead of loading separate all-MiniLM-L6-v2 (~80MB savings)
- [ ] Batch multiple chunks into single LLM enrichment calls
- [ ] Use structured output / JSON mode instead of fragile code fence parsing
- [ ] Multilingual NER model (spaCy en_core_web_sm is English-only)

## Frontend
- [ ] Full UI rework
- [ ] DMG packaging via Tauri sidecar (bundle Python backend)

## Done
- [x] Configurable host/port from config.yaml
- [x] CORS regex instead of hardcoded origin list
- [x] Chat route refactor: extracted shared helpers, dropped WebSocket endpoint
- [x] Inline imports cleanup in embed.py, search.py, store.py
- [x] Removed all truncations in reranker, LLM enrichment, and contextual prefixes
- [x] MCP server: 7 tools mounted at /mcp (streamable HTTP) + stdio via --mcp-stdio
- [x] Added expand parameter to SearchEngine.search()
- [x] Added get_chunk_by_id() to Store
- [x] EPUB extractor: recursive heading detection + code block preservation
- [x] PDF extractor: font-aware code detection + bold heading detection, OCR disabled
- [x] Provider cleanup: deleted CLI providers, simplified to custom/OpenRouter only
- [x] Default model set to openrouter/elephant-alpha (free, 100B, tool use)
- [x] Source-specific location fields: page_num, section_heading, chapter, line_start, line_end, chunk_index
- [x] Document expansion via chunk_index instead of broken timestamp-based expansion
- [x] Full ingestion pipeline tested end-to-end (EPUB -> chunks -> search -> RAG chat)
