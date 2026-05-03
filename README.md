# Lore

**An MCP knowledge base that turns scattered docs, papers, and code into agent-ready context with enrichment, cross-source linking, and wiki synthesis.**

Most RAG systems stop at semantic search over a folder. That's not enough for real agent workflows. Agents need to find the right source, understand what kind of source it is, connect related material across collections, and return context that is traceable and usable. Lore is built for that layer: a local-first research interface that helps agents search, enrich, trace, and synthesize knowledge across mixed content.

## Demo

<!-- TODO: Add GIF/screenshots showing multi-source search, cross-source discovery, and wiki synthesis -->

*Search docs, papers, wiki pages, and enriched sources through one MCP interface.*

## Key capabilities

- **Hybrid search** across PDFs, EPUBs, doc sites, videos, code, and web pages with six scoring signals and cross-encoder reranking
- **4-stage enrichment** that adds titles, summaries, concept tags, entity extraction, importance scores, and section-level synthesis to every chunk
- **Wiki synthesis** that generates entity, concept, and comparison pages with claim-level provenance and verification status
- **Cross-source intelligence** via fuzzy entity merging, co-occurrence graphs, and Jaccard similarity indexes that surface connections between unrelated sources
- **23 MCP tools** designed for progressive disclosure: compact search results, expandable context, wiki pages, graph exploration, recursive generation, and health auditing

## Quick setup

```bash
git clone https://github.com/sidskarkii/Lore.git
cd Lore
./scripts/setup.sh

# Register with Claude Code
claude mcp add lore -- /path/to/Lore/.venv/bin/python -m lore --mcp-stdio
```

## Architecture

```
Agent (Claude Code, Cursor, etc.)
      |
      | MCP (stdio)
      v
+-----------+     +-----------+     +-----------+
|  Search   |     |  Enrich   |     |   Wiki    |
|  Engine   |     |  Pipeline |     |   Layer   |
+-----+-----+     +-----+-----+     +-----+-----+
      |                 |                 |
      v                 v                 v
+-----------+     +-----------+     +-----------+
|  LanceDB  |     |  Archive  |     |  SQLite   |
|  vectors  |     |  per-src  |     |  events   |
+-----------+     +-----------+     +-----------+
```

## How it works

**Search.**
Queries hit six signals in parallel: vector similarity (EmbeddingGemma ONNX), BM25 full-text, entity boost through a fuzzy entity index, reciprocal rank fusion, FlashRank cross-encoder reranking, and query intent detection that routes wiki vs chunk results. Post-reranking applies Wilson Score from interaction history, importance boosts, and session-aware dedup.

**Enrichment.**
Every ingested source goes through a 4-stage pipeline. Stage 1 extracts keywords (KeyBERT) and entities (spaCy). Stage 2 uses an LLM with a rolling key dictionary to generate titles, summaries, concept tags, and importance scores for each chunk. Stage 3 synthesizes section-level summaries with a concept ledger that tracks how ideas evolve. Stage 4 produces a book/document-level summary. Each stage works from original text, not prior stage output.

**Wiki layer.**
Lore generates synthesized wiki pages (entity, concept, source, comparison) that aggregate evidence across all sources. Each claim stores which chunks support it, from how many sources, and gets a verification status. Contradiction detection uses embedding similarity and negation asymmetry to find cross-page conflicts. Recursive generation discovers missing pages and ranks candidates by link pressure, evidence count, source diversity, and graph centrality.

**Cross-source synthesis.**
A fuzzy entity index (Jaro-Winkler, type-aware thresholds) links mentions across sources automatically. Entity and keyword co-occurrence graphs (NPMI-weighted, Louvain communities) reveal topic structure. When new content is ingested, overlapping wiki pages are proactively regenerated so knowledge compounds immediately.

## Supported sources

| Source | Extraction | Notes |
|--------|-----------|-------|
| PDF | pymupdf, font-aware heading detection | Chapter pattern fallback, heading validation |
| EPUB | Spine-based, recursive headings | EPUB3 compatible |
| YouTube | yt-dlp subtitles + metadata | Chapters map to sections, tags to keywords |
| Audio/Video | sherpa-onnx Whisper medium.en | ONNX Runtime, auto-downloads |
| Web pages | trafilatura | Article content extraction |
| Markdown/MDX | Direct chunking | Doc site ingestion with smart file filtering |
| Code | Language-aware chunking | .py, .js, .ts, .java, .go, .rs, etc. |

## LLM provider

> **Note:** Enrichment and wiki generation make multiple LLM calls per file. A large PDF can use 50k+ tokens. Use a free or low-cost provider.

Configure in `.env` at the project root:

```bash
LORE_CUSTOM_BASE_URL=https://integrate.api.nvidia.com/v1
LORE_CUSTOM_API_KEY=your-api-key-here
LORE_CUSTOM_MODEL=nvidia/nemotron-3-super-120b-a12b
```

Works with any OpenAI-compatible endpoint: [Nvidia NIM](https://build.nvidia.com/) (free tier), [OpenRouter](https://openrouter.ai/), [Ollama](https://ollama.com/) (local), or any self-hosted API.

If no provider is configured and Lore runs under Claude Code, it automatically uses the **Claude CLI provider**, inheriting your subscription. Zero-config but slower.

Core search works without an LLM. Only enrichment and wiki generation need one.

## Data layout

```
~/.lore/
    store/              LanceDB vector store
    archive/            Per-source: meta.json, chunks.json, summaries
    wiki/               Pages, manifests, staleness tracking
    models/             Auto-downloaded reranker + transcription models
    app.db              SQLite (interactions, events, ingestion log)
```

## Tech stack

Python 3.10+, LanceDB, ONNX Runtime, FlashRank, sherpa-onnx, SQLite, KeyBERT, spaCy, rapidfuzz, networkx, MCP SDK, FastAPI, PyMuPDF, trafilatura. All models auto-download on first use. Runs on CPU with Apple Silicon optimization.
