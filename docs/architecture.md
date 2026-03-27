# TutorialVault Architecture

> A universal knowledge base that turns videos, docs, and playlists into a searchable, chat-ready library.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Desktop App (Tauri v2 + Svelte + TypeScript)               │
│  ┌──────────┬──────────┬─────────┬─────────┬──────────┐     │
│  │  Chat    │Collection│ Ingest  │ Explore │ Settings │     │
│  │  Tab     │  Tab     │  Tab    │  Tab    │  Tab     │     │
│  └──────┬───┴────┬─────┴────┬────┴────┬────┴────┬─────┘     │
│         │ HTTP / WebSocket  │         │         │           │
└─────────┼───────────────────┼─────────┼─────────┼───────────┘
          │                   │         │         │
┌─────────▼───────────────────▼─────────▼─────────▼───────────┐
│  Python Backend (FastAPI)                                    │
│                                                              │
│  ┌─ API Layer ──────────────────────────────────────────┐    │
│  │  /api/health    /api/search    /api/chat             │    │
│  │  /api/providers /api/collections /api/sessions       │    │
│  │  Thin routes — validate input, call core, return     │    │
│  └──────────────────────┬───────────────────────────────┘    │
│                         │                                    │
│  ┌─ Core Engine ────────▼───────────────────────────────┐    │
│  │                                                      │    │
│  │  Search Pipeline:                                    │    │
│  │  Query → Embed → Vector Search ──┐                   │    │
│  │                   BM25 Search ───┤ RRF Fusion        │    │
│  │                                  ↓                   │    │
│  │                          FlashRank Rerank            │    │
│  │                                  ↓                   │    │
│  │                      Parent Window Expand            │    │
│  │                                  ↓                   │    │
│  │                          Ranked Results              │    │
│  │                                                      │    │
│  │  Storage:                                            │    │
│  │  ┌─────────────┐  ┌──────────────┐                   │    │
│  │  │  LanceDB    │  │   SQLite     │                   │    │
│  │  │  (vectors)  │  │  (sessions)  │                   │    │
│  │  └─────────────┘  └──────────────┘                   │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─ Providers ──────────────────────────────────────────┐    │
│  │  Claude Code │ OpenCode │ Codex │ Kilo │ Custom      │    │
│  │  (5.2s)      │ (4.4s)   │(2.6s) │(20s) │ (any URL)  │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. No PyTorch

The entire backend runs without PyTorch. Embedding and reranking use ONNX Runtime. Base install is ~100MB instead of ~2.2GB.

| Component | Traditional | Ours |
|-----------|------------|------|
| Embedding | sentence-transformers (pulls torch, ~2GB) | ONNX Runtime + EmbeddingGemma-300M q4 (~238MB) |
| Reranking | CrossEncoder (pulls torch) | FlashRank MiniLM-L12 (~34MB built-in ONNX) |
| Runtime | PyTorch (~2GB) | onnxruntime (~50MB) |

### 2. Own Everything

No LangChain, no LlamaIndex, no black-box orchestration frameworks. Every component is our code:

- Chunking logic — overlapping time windows
- Embedding — direct ONNX Runtime session management
- Vector search — LanceDB queries with filtering
- Hybrid search — RRF fusion implementation
- Reranking — FlashRank integration
- Provider abstraction — subprocess wrappers for each CLI

The code is readable because there's no framework abstractions to unwrap.

### 3. Zero Mandatory Cost

Every user can run the full app without paying anything:

- **Embedding:** runs locally, no API
- **Reranking:** runs locally, no API
- **Vector DB:** LanceDB is embedded, no server
- **LLM:** OpenCode (6 free models) or Kilo (10 free models)

Users with existing subscriptions (Claude Max, ChatGPT Plus) can use those too.

### 4. Clean Separation

```
src/tutorialvault/
├── core/        # Business logic — no HTTP, no CLI, no UI
├── providers/   # CLI wrappers — no HTTP, no storage
├── api/         # HTTP routes — no logic, just validation + routing
└── ui/          # Desktop app — consumes API only
```

The API routes are intentionally thin. A route validates the request, calls a function in `core/` or `providers/`, and returns the response. No business logic lives in the routes.

## Component Details

### Embedding: EmbeddingGemma-300M

Google's EmbeddingGemma-300M, #1 on MTEB for models under 500M parameters. We use the q4 quantized ONNX export (188MB).

Evaluated against BGE-M3 (the standard recommendation):
- Same accuracy on our test set (5/5)
- 6x smaller on disk (188MB vs 1.2GB fp32)
- 14ms per text on CPU
- No PyTorch required

The model downloads on first use from HuggingFace. Only the q4 variant files are fetched, not the full repository.

### Reranking: FlashRank MiniLM-L12

Cross-encoder reranker that re-scores candidates after initial retrieval. We tested 6 rerankers on 211,000 real chunks from our Blender/Houdini tutorial corpus:

| Reranker | Size | Speed | Quality |
|----------|------|-------|---------|
| FlashRank TinyBERT | 4MB | 11ms | Decent |
| **FlashRank MiniLM-L12** | **34MB** | **349ms** | **Best** |
| GTE-reranker int8 | 341MB | 1,693ms | Mixed |
| Qwen3-Reranker-0.6B | 1.19GB | 10,262ms | Worse on our domain |

MiniLM-L12 won despite being 35x smaller than Qwen3-Reranker. Domain-specific data matters more than benchmark scores.

### Search Pipeline

```
Query
  ↓
Embed (EmbeddingGemma q4, ~14ms)
  ↓
┌─────────────────────────────────┐
│ Vector Search    BM25 Search    │  (parallel, 30 candidates each)
│ (cosine sim)    (keyword match) │
└────────┬────────────┬───────────┘
         ↓            ↓
     RRF Fusion (merge ranked lists)
         ↓
     Top N×4 candidates
         ↓
     FlashRank Rerank (~349ms)
         ↓
     Top N results
         ↓
     Parent Window Expansion (±2.5 min context)
         ↓
     Final results with timestamps + metadata
```

**RRF (Reciprocal Rank Fusion):** Merges multiple ranked lists using `score = Σ 1/(k + rank)`. Simple, effective, no learned parameters.

**Parent Window Expansion:** When a chunk matches, we grab all chunks within ±2.5 minutes of it from the same episode. This gives the LLM more context than the narrow matched chunk alone.

### Storage

**LanceDB** for vectors — embedded database, no server process, data is a folder on disk. Supports vector search, full-text search (BM25), and SQL-like filtering.

**SQLite** for chat history — stdlib (`import sqlite3`), zero dependencies. Tables: `chat_sessions`, `messages` (with FTS5 full-text search index), `settings`. WAL mode for concurrent reads.

Both are single-file databases. Backing up the app means copying two files.

### Providers

Four CLI tools wrapped as LLM backends, plus a custom endpoint option:

| Provider | How it works | Auth model |
|----------|-------------|------------|
| Claude Code | `claude -p` with isolated config dir | OAuth (Max/Pro sub) |
| OpenCode | `opencode run --format json` | Free models, no auth needed |
| Codex | `codex exec --json --full-auto` | OAuth (ChatGPT sub) |
| Kilo | `kilo --auto --json --nosplash` | Auto-auth, free models |
| Custom | OpenAI Python SDK | User provides URL + key |

Each provider handles detection, authentication, JSONL parsing, and error handling independently. The registry manages switching between them.

### API

16 endpoints, all documented with OpenAPI (Swagger UI at `/api/docs`):

| Category | Endpoints |
|----------|-----------|
| Health | `GET /api/health` |
| Search | `POST /api/search` |
| Chat | `POST /api/chat`, `WS /api/chat/stream` |
| Providers | `GET /api/providers`, `POST /api/providers/active`, `POST /api/providers/install`, `POST /api/providers/authorize`, `POST /api/providers/test` |
| Collections | `GET /api/collections`, `DELETE /api/collections` |
| Sessions | `GET /api/sessions`, `GET /api/sessions/{id}`, `PATCH /api/sessions/{id}`, `DELETE /api/sessions/{id}`, `POST /api/sessions/search` |

All request/response payloads are defined as Pydantic models in `schemas.py`. The frontend TypeScript types should mirror these exactly.

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Desktop | Tauri v2 | ~10MB installer vs Electron's 200MB, Rust runtime |
| Frontend | Svelte + TypeScript | Smallest bundle, fastest, least boilerplate |
| Backend | Python FastAPI | Async, auto-docs, Pydantic validation |
| Embedding | ONNX Runtime | No PyTorch, 50MB |
| Reranking | FlashRank | No PyTorch, 34MB |
| Vector DB | LanceDB | Embedded, no server |
| Chat DB | SQLite | Stdlib, zero deps |
| Communication | HTTP + WebSocket | Standard, no custom protocols |

## What's Built vs Planned

| Phase | Status | What |
|-------|--------|------|
| 1. Core Engine | ✅ Done | Config, chunk, embed, store, search |
| 2. Server + Providers | ✅ Done | FastAPI, 4 CLI providers, SQLite persistence |
| 3. Desktop App | Planned | Tauri + Svelte (setup wizard, chat, collections, ingest, explore, settings) |
| 4. Ingestion Pipeline | Planned | Whisper transcription, YouTube download, doc ingestion |
| 5. Local GGUF | Planned | llama-cpp-python, own inference server |
| 6. Fine-tuning | Future | Synthetic Q&A, QLoRA, 3-way eval |
