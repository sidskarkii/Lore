# Lore

**MCP-native universal knowledge base with multi-stage enrichment and cross-source intelligence.**

Lore turns any content - books, docs, videos, code, web pages - into a searchable, enriched knowledge layer that AI agents plug into natively via MCP. It doesn't just embed and retrieve. It extracts entities, generates concept tags, builds section summaries, tracks cross-source connections, and learns from agent interactions over time.

Built for macOS Apple Silicon. Local-first: ONNX embeddings, local transcription, no cloud required for core functionality.

## Why this exists

RAG systems typically stop at "chunk text, embed, retrieve." Lore goes further:

- **Multi-stage enrichment** - each chunk gets a title, summary, concept tags, importance score, confidence level, hypothetical questions, and self-containedness flag. Section and book-level summaries synthesize across chunks. A rolling key dictionary (inspired by [MDKeyChunker](https://arxiv.org/abs/2603.23533)) keeps concept tags consistent across hundreds of chunks.

- **Cross-source entity discovery** - a fuzzy entity index (Jaro-Winkler similarity, type-aware merging, adaptive thresholds) automatically finds connections between sources. An entity mentioned in a strategy book links to the same entity in a psychology book without manual tagging.

- **Agent-native design** - 12 MCP tools with progressive disclosure. Compact search results (~50 tokens each) let agents scan before committing context. Built-in dedup, session-aware scoring with TTL, and interaction logging that feeds back into ranking via Wilson Score.

- **Content-agnostic** - same pipeline handles PDFs, EPUBs, YouTube videos (with chapter mapping), audio transcription (sherpa-onnx Whisper), web pages, and code. Not book-specific.

## Architecture

```
Agent (Claude Code, Cursor, etc.)
    |
    | MCP (stdio / HTTP)
    v
+-------------------------------------------+
|            Lore MCP Server                |
|                                           |
|  12 tools: intro, search, search_deep,    |
|  get_context, get_toc, find_related,      |
|  entity_index, reset_session, ingest,     |
|  ingest_status, rate_result,              |
|  delete_collection                        |
+-------------------+-----------------------+
                    |
       +------------+------------+
       v            v            v
  +---------+  +---------+  +---------+
  | Search  |  | Enrich  |  | Entity  |
  | Pipeline|  | Pipeline|  |  Index  |
  +----+----+  +----+----+  +----+----+
       |            |            |
       v            v            v
  +---------+  +---------+  +---------+
  | LanceDB |  | Archive |  | SQLite  |
  | vectors |  | per-src |  | ratings |
  +---------+  +---------+  +---------+
```

## Search pipeline

Five-signal hybrid retrieval:

1. **Vector similarity** - EmbeddingGemma-300M (ONNX, 188MB, q4 quantized)
2. **BM25** - LanceDB full-text search
3. **Entity boost** - query entities expanded through fuzzy entity index, matched against chunk entities + keywords + concept tags
4. **Reciprocal Rank Fusion** - merges all three signals
5. **Cross-encoder reranking** - FlashRank ms-marco-MiniLM-L-12-v2

Post-reranking adjustments:
- **Wilson Score** from interaction history (fetches, ignores, explicit ratings)
- **Importance boost** from enrichment (1-5 scale)
- **Session deprioritization** with 30-min TTL (fetched chunks scored lower, expire back to full relevance)
- **Built-in dedup** - `get_context` never returns chunks the agent already has

## Enrichment pipeline

Research-backed, four-stage pipeline. Each stage gets original text - no telephone game.

| Stage | Method | Output |
|-------|--------|--------|
| **1. Classical ML** | KeyBERT + spaCy NER | Keywords, named entities |
| **2. Chunk enrichment** | LLM with rolling key dict | Title, summary, tags, concept tags, importance (1-5 with rubric), questions, confidence, self-contained flag - 10 fields |
| **3. Section synthesis** | LLM with concept ledger | Section summary, themes, notable points, tensions. Ledger tracks concept evolution (keep/refine/add/downweight) |
| **4. Book summary** | LLM with concept aggregation | Overview, themes, takeaways, cross-section patterns |

**Rolling key dictionary** (MDKeyChunker pattern): as chunks are processed, an accumulated dictionary of concept tags is passed to each subsequent chunk. The LLM reuses existing tags when applicable instead of inventing synonyms. Produces ~90% tag reuse rate across a document.

**Concept ledger**: Stage 3 maintains a per-section ledger of concepts with explicit actions - an author contradicting an earlier claim gets `downweight`, a reinforced concept gets `refine`. This produces section summaries that reflect the actual arc of an argument.

## Entity index

Fuzzy entity merging across all collections:

- **Normalization** - NFKD unicode, strip possessives/articles, control characters
- **Noise filtering** - structural references (chapter numbers, figure labels), generic words
- **Type correction** - conservative gazetteer for known countries/cities (France typed as PERSON -> GPE)
- **Adaptive thresholds** - single-token <=5 chars: exact match only. 6-8 chars: 95%. Multi-token: 90%
- **Type-aware merging** - PERSON and GPE clusters never merge

Cross-source entities surface automatically: a person mentioned in a strategy book connects to the same person in a psychology book without manual linking.

## Agent experience

Three-layer progressive disclosure:

1. **On connect** - dynamic MCP instructions with chunk counts, topics, numbered retrieval workflow with token costs
2. **`intro` tool** - full orientation: collection summaries with themes, cross-source entities, usage stats, suggested workflows
3. **Tool docstrings** - step numbers matching the retrieval workflow ("Step 1: search", "Step 3: get_context")

Session intelligence:
- Interaction logging captures every search, fetch, and rating
- Wilson Score confidence intervals adjust rankings over time
- TTL-based re-eligibility (chunks become full-score after 30 min)
- `reset_session` for immediate refresh after context compaction

## Supported sources

| Source | Extraction | Enrichment |
|--------|-----------|------------|
| PDF | pymupdf, font-aware heading detection, chapter pattern fallback, heading validation | Full 4-stage pipeline |
| EPUB | Spine-based, recursive headings | Full 4-stage pipeline |
| YouTube | yt-dlp subtitles, chapters -> section headings, tags -> keywords | Full pipeline + video metadata |
| Audio/Video | sherpa-onnx Whisper medium.en (~1.5GB, auto-downloads, ONNX Runtime) | Transcription + pipeline |
| Web pages | trafilatura | Full pipeline |
| Markdown/text | Direct chunking | Full pipeline |

## Setup

```bash
git clone https://github.com/sidskarkii/Lore.git
cd Lore
./scripts/setup.sh
```

One command: venv, all deps, spaCy model, system tools (yt-dlp, ffmpeg via brew), `.env` template, full verification.

```bash
# Register with Claude Code
claude mcp add lore -- /path/to/Lore/.venv/bin/python -m lore --mcp-stdio

# Or run as HTTP server
python -m lore  # MCP at http://localhost:52105/mcp
```

## Tech

Python, LanceDB, ONNX Runtime, FlashRank, sherpa-onnx, SQLite, KeyBERT, spaCy, rapidfuzz, MCP SDK, FastAPI

All models download automatically on first use. No GPU required - runs on CPU with Apple Silicon optimization.

## Data

```
~/.lore/
    store/              LanceDB vector store
    archive/            Per-source: meta.json, extracted.md, chunks.json, summaries
    models/             Reranker + transcription models
    app.db              SQLite (interactions, ratings)
    entity_index.json   Fuzzy-merged entity clusters
```

Archive means you can wipe the vector store and rebuild without re-calling LLMs.
