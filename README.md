# Lore

A local-first knowledge base that turns videos, documents, and playlists into a searchable, chat-ready library. You feed it content (YouTube videos, local files, PDFs, web pages) and it transcribes, chunks, embeds, and indexes everything. Then you search or chat with it and get source-grounded answers that link back to exact timestamps in the original material.

## What it does

**Ingest** any combination of:
- YouTube videos and playlists
- Local video/audio files (mp4, mkv, webm, etc.)
- Documents (PDF, EPUB, Markdown, plain text)
- Source code (Python, JS, Go, Rust, etc.)
- Web pages

**Search** using hybrid retrieval:
- Vector similarity (EmbeddingGemma-300M, ONNX)
- Full-text search (BM25 via SQLite FTS5)
- Reciprocal Rank Fusion to merge both result sets
- Cross-encoder reranking (FlashRank) for final ordering
- Multi-hop decomposition for complex questions

**Chat** with your knowledge base:
- RAG pipeline retrieves relevant chunks before answering
- Responses cite sources with timestamps you can click
- Streaming via Server-Sent Events
- Persistent chat sessions stored in SQLite

**Use any LLM** you want:
- Ollama, LM Studio, or any OpenAI-compatible endpoint
- OpenRouter for cloud models
- Local GGUF files via llama-cpp-python
- Switch providers from the UI without restarting

## Architecture

```
React + Tauri (desktop app)
        |
        | HTTP / SSE
        v
FastAPI (Python backend, port 8000)
        |
        +-- Embedding:     EmbeddingGemma-300M via ONNX Runtime (no PyTorch)
        +-- Vector store:  LanceDB
        +-- Reranker:      FlashRank (ms-marco-MiniLM-L-12-v2)
        +-- Sessions:      SQLite with WAL mode
        +-- Transcription:  faster-whisper (optional)
        +-- LLM:           pluggable provider system
```

The backend does all the heavy lifting. The frontend is a React app wrapped in Tauri for a native desktop experience. In development you can also just run the frontend in a browser.

## Project structure

```
src/lore/
    api/            FastAPI routes (chat, search, ingest, sessions, providers, etc.)
    core/           Business logic (embedding, chunking, search, ingestion, transcription)
    providers/      LLM provider implementations (custom, local GGUF, etc.)

ui/
    src/            React + TypeScript frontend
    src-tauri/      Tauri (Rust) desktop wrapper
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- An LLM provider (Ollama running locally is the easiest option)

For video transcription (optional):
- ffmpeg installed and on PATH

For the desktop app (optional):
- Rust toolchain (rustup)

## Setup

### Backend

```bash
cd tutorialvault

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate

# Install core dependencies
pip install -e .

# (Optional) Install transcription support
pip install -e ".[transcribe]"

# (Optional) Install local GGUF model support
pip install -e ".[local]"
```

On first run, the embedding model (~188MB) and reranker (~34MB) are downloaded automatically from HuggingFace.

### Frontend

```bash
cd ui
npm install
```

### LLM provider

The easiest way to get started is with Ollama:

```bash
# Install Ollama (https://ollama.com), then:
ollama pull llama3.2
```

Then create a `config.local.yaml` in the project root:

```yaml
provider:
  active: custom
  custom:
    base_url: http://localhost:11434/v1
    api_key: ollama
    model: llama3.2
```

This file is gitignored. You can point it at any OpenAI-compatible API (OpenRouter, LM Studio, vLLM, etc.) by changing the base_url, api_key, and model.

## Running

Start both the backend and frontend:

```bash
# Terminal 1: backend
source .venv/bin/activate
python -m lore
# API runs on http://127.0.0.1:8000
# Docs at http://127.0.0.1:8000/api/docs

# Terminal 2: frontend (browser mode)
cd ui
npm run dev
# Opens on http://localhost:1420
```

To run as a desktop app instead:

```bash
cd ui
npm run tauri dev
```

## Usage

1. Open the app in your browser (localhost:1420) or as a desktop app
2. Go to the "Add" tab in the sidebar
3. Ingest some content: paste a YouTube URL, point it at a folder of videos, or upload documents
4. Switch to the "Chat" tab and ask questions about your content
5. Click on source citations to jump to the exact timestamp or section

## API

The backend exposes a REST API. Full interactive docs are available at `/api/docs` when the server is running.

Key endpoints:

```
GET   /api/health              Server status and stats
POST  /api/search              Hybrid search over indexed content
POST  /api/chat/stream         Chat with streaming (SSE)
POST  /api/ingest/youtube      Ingest a YouTube video or playlist
POST  /api/ingest/file         Ingest a local file
POST  /api/ingest/url          Ingest a web page
GET   /api/collections         List all indexed collections
GET   /api/sessions            List chat sessions
GET   /api/providers           List available LLM providers
```

## Configuration

`config.yaml` has the defaults for embedding, chunking, search, and transcription. You generally don't need to touch it.

`config.local.yaml` is where you put provider credentials and overrides. It is not checked into git.

## Tech stack

**Backend:** Python, FastAPI, LanceDB, ONNX Runtime, FlashRank, SQLite, faster-whisper

**Frontend:** React, TypeScript, Vite, Tailwind CSS, Zustand, Tauri

**Models (downloaded automatically):** EmbeddingGemma-300M (ONNX, 188MB), ms-marco-MiniLM-L-12-v2 (34MB)
