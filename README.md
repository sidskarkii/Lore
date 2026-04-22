# Lore

A local-first knowledge base that AI agents plug into via MCP. Feed it books, documents, videos, code, and web pages. It extracts, chunks, enriches, and indexes everything. Agents search it, browse it, and get smarter over time through interaction logging and relevance feedback.

Not a wrapper around a vector DB. A multi-stage enrichment pipeline that produces chunk titles, section summaries, book overviews, concept tags, and importance scores. Every search result carries enough metadata for an agent to decide what to read without reading it.

## How it works

```
Agent (Claude Code, Cursor, etc.)
    |
    | MCP (stdio or HTTP)
    v
Lore MCP Server
    |
    +-- Search:       vector + BM25 + entity boost + cross-encoder reranking
    +-- Embedding:    EmbeddingGemma-300M (ONNX, no PyTorch for inference)
    +-- Enrichment:   KeyBERT + spaCy NER + multi-stage LLM pipeline
    +-- Vector store: LanceDB at ~/.lore/store/
    +-- Metadata:     SQLite at ~/.lore/app.db (interactions, ratings)
    +-- Archive:      ~/.lore/archive/ (per-source extracted text + enrichment)
```

## Setup

```bash
# Clone and install
git clone https://github.com/sidskarkii/Lore.git
cd Lore
python -m venv .venv
source .venv/bin/activate
pip install -e ".[enrich]"
python -m spacy download en_core_web_sm

# (Optional) Set up an LLM provider for enrichment
# Create .env in project root:
echo "LORE_CUSTOM_API_KEY=your-openrouter-key" > .env

# Register with Claude Code
claude mcp add lore -- $(pwd)/.venv/bin/python -m lore --mcp-stdio
```

On first run, the embedding model (~188MB) and reranker (~34MB) download automatically from HuggingFace.

Enrichment works without an LLM provider (keywords and entities via KeyBERT + spaCy). Add an OpenRouter API key for full enrichment (titles, summaries, concept tags, section summaries, book overviews).

## MCP tools

| Tool | What it does |
|------|-------------|
| `search` | Hybrid retrieval. Compact results by default (metadata only). Entity-boosted via NER |
| `search_deep` | Multi-hop: decomposes complex queries into sub-queries via LLM |
| `get_context` | Paginated expansion around a search result. Agent controls page size |
| `get_toc` | Table of contents for a collection. Sections with chunk counts and token estimates |
| `ingest` | Non-blocking. Queues the job, returns a job ID immediately |
| `ingest_status` | Check progress: "Stage 2: chunk titles batch 3/7 (0 cached)" |
| `list_collections` | What's indexed, with topics and episode counts |
| `delete_collection` | Remove a collection |
| `rate_result` | Explicit feedback. Improves future rankings over time |
| `health` | Server status, chunk count, active models |

## Enrichment pipeline

Ingestion runs a multi-stage pipeline. Each stage gets original text, not distilled-from-distilled.

**Stage 1 -- Classical ML (no LLM)**
Keywords via KeyBERT, named entities via spaCy NER.

**Stage 2 -- Chunk-level enrichment (LLM)**
Title, summary, topic tags, concept tags, importance score (1-5). Prompt includes book title, section heading, and chapter name for context.

Concept tags are specific enough to bridge across books: `deception-as-advantage`, `reciprocity-principle`, `perception-control`. These power cross-source discovery.

**Stage 3 -- Section summaries (LLM)**
Progressive passes over full original text (5000 tokens per pass with running summary). Produces section summary, key concepts, key entities.

**Stage 4 -- Book summary (LLM)**
Reads all section summaries + table of contents. Produces overview, main themes, key takeaways, book-level tags.

All enrichment is archived per-source at `~/.lore/archive/{collection}/` with `meta.json`, `extracted.md`, `chunks.json`, `section_summaries.json`, and `book_summary.json`.

## Search

Hybrid retrieval pipeline:
1. Vector similarity (EmbeddingGemma-300M, ONNX)
2. BM25 full-text search (LanceDB FTS)
3. Entity-boosted ranking (spaCy extracts entities from query, matches against stored entities + keywords + concept tags)
4. Reciprocal Rank Fusion merges all three signals
5. Cross-encoder reranking (FlashRank ms-marco-MiniLM-L-12-v2)

Results are compact by default: chunk ID, score, token count, title, summary, concept tags, importance, location. No full text unless requested. Agent scans metadata, fetches what it needs via `get_context`.

## Chunk IDs

Every token in a chunk ID is meaningful:

| Source | Example |
|--------|---------|
| EPUB | `ego_is_the_enemy_ch_the_painful_prologue_0004` |
| PDF | `the_art_of_war_i_laying_plans_0005` |
| Video | `blender_tutorial_uv_unwrapping_basics_t04m32s` |
| Code | `lore_codebase_src_lore_core_search_py_L45-90` |
| Web | `karpathy_wiki_career_and_research_0002` |

## Data

All data lives at `~/.lore/` (override with `LORE_DATA_DIR` env var):

```
~/.lore/
    store/          LanceDB vector store
    archive/        Per-source: meta.json, extracted.md, chunks.json, summaries
    app.db          SQLite (interaction logs, chunk ratings)
```

Archive means you can wipe the vector store and rebuild from archived enrichment without re-calling LLMs.

## Configuration

`config.yaml` in the project root has defaults for embedding, chunking, search, and transcription.

`config.local.yaml` (gitignored) for provider overrides:

```yaml
provider:
  active: custom
  custom:
    base_url: https://openrouter.ai/api/v1
    model: openai/gpt-oss-120b:free
```

Or use environment variables: `LORE_CUSTOM_BASE_URL`, `LORE_CUSTOM_API_KEY`, `LORE_CUSTOM_MODEL`.

Works with any OpenAI-compatible API: OpenRouter, Ollama, LM Studio, Together, Groq.

## Running

**As MCP server (recommended):**
```bash
claude mcp add lore -- /path/to/Lore/.venv/bin/python -m lore --mcp-stdio
```
Auto-starts when the agent connects. Zero manual server management.

**As HTTP server (for multi-agent or remote access):**
```bash
python -m lore
# MCP endpoint at http://localhost:52105/mcp
```

## Tech stack

Python, FastAPI, LanceDB, ONNX Runtime, FlashRank, SQLite, KeyBERT, spaCy, MCP SDK

Models (downloaded automatically): EmbeddingGemma-300M (ONNX, 188MB), ms-marco-MiniLM-L-12-v2 (34MB), all-MiniLM-L6-v2 (KeyBERT, 80MB), en_core_web_sm (spaCy, 12MB)

## Target

macOS, Apple Silicon (M1+), 8GB+ RAM
