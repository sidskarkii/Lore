# API Reference

> TutorialVault backend API. Start the server with `python -m tutorialvault` (default: `http://localhost:8000`).
>
> Interactive docs available at `/api/docs` (Swagger) and `/api/redoc`.

## Health

### `GET /api/health`

Server status, model info, and basic stats.

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "embedding_model": "onnx-community/embeddinggemma-300m-ONNX",
  "reranker_model": "ms-marco-MiniLM-L-12-v2",
  "total_chunks": 8,
  "active_provider": "kilo"
}
```

---

## Search

### `POST /api/search`

Hybrid search (vector + BM25) with RRF fusion and cross-encoder reranking.

**Request:**
```json
{
  "query": "how to rig a character in blender",
  "n_results": 5,
  "topic": "3d",
  "subtopic": "blender"
}
```

`topic` and `subtopic` are optional filters.

**Response:**
```json
{
  "query": "how to rig a character in blender",
  "total": 5,
  "results": [
    {
      "text": "The expanded chunk text with surrounding context...",
      "collection": "blender_rigging_101",
      "collection_display": "Blender Rigging 101",
      "episode_num": 3,
      "episode_title": "Creating the Armature",
      "timestamp": "05:30",
      "start_sec": 330,
      "end_sec": 630,
      "url": "https://youtube.com/watch?v=...",
      "topic": "3d",
      "subtopic": "blender"
    }
  ]
}
```

---

## Chat

### `POST /api/chat`

Search for relevant sources, then send the query + sources to the active LLM provider.

**Request:**
```json
{
  "session_id": null,
  "messages": [
    {"role": "user", "content": "How do I rig a character in Blender?"}
  ],
  "n_sources": 5,
  "topic": null,
  "subtopic": null,
  "model": null,
  "provider": null
}
```

- `session_id`: Pass null to create a new session, or an existing ID to continue.
- `provider`: Override the active provider (e.g. `"claude_code"`).
- `model`: Override the provider's default model.

**Response:**
```json
{
  "session_id": "a1b2c3d4e5f6",
  "answer": "To rig a character in Blender, first create an armature...",
  "sources": [
    {
      "text": "...",
      "collection_display": "Blender Rigging 101",
      "episode_title": "Creating the Armature",
      "timestamp": "05:30",
      "start_sec": 330,
      "end_sec": 630,
      "url": "https://youtube.com/watch?v=..."
    }
  ],
  "provider": "kilo",
  "model": "kilo-auto/free"
}
```

Messages are persisted to SQLite. The session can be resumed by passing its `session_id` in the next request.

### `WS /api/chat/stream`

WebSocket endpoint for streaming chat responses. Send a ChatRequest JSON, receive events:

```json
{"type": "session", "data": "a1b2c3d4e5f6"}
{"type": "source",  "data": {"collection_display": "...", "timestamp": "05:30", ...}}
{"type": "source",  "data": {"collection_display": "...", "timestamp": "12:15", ...}}
{"type": "token",   "data": "To "}
{"type": "token",   "data": "rig "}
{"type": "token",   "data": "a character..."}
{"type": "done",    "data": "a1b2c3d4e5f6"}
```

Error events: `{"type": "error", "data": "error message"}`.

---

## Providers

### `GET /api/providers`

List all providers with installation status, auth, and available models.

**Response:**
```json
{
  "providers": [
    {
      "name": "claude_code",
      "display_name": "Claude Code",
      "installed": true,
      "authenticated": true,
      "version": "2.1.83 (Claude Code)",
      "user": "user@email.com",
      "error": null,
      "models": [
        {"id": "sonnet", "name": "Claude Sonnet", "free": false, "context_window": 200000}
      ],
      "free_model_count": 0,
      "install_command": "npm install -g @anthropic-ai/claude-code",
      "is_active": false
    },
    {
      "name": "kilo",
      "display_name": "Kilo CLI",
      "installed": true,
      "authenticated": true,
      "version": "0.26.1",
      "models": [
        {"id": "kilo-auto/free", "name": "Kilo Auto Free", "free": true, "context_window": 204800}
      ],
      "free_model_count": 10,
      "is_active": true
    }
  ],
  "active": "kilo"
}
```

### `POST /api/providers/active`

Switch the active provider.

```json
{"provider": "claude_code"}
```

### `POST /api/providers/install`

Install a provider's CLI tool. The frontend should confirm with the user before calling.

```json
{"provider": "kilo"}
```

### `POST /api/providers/authorize`

For providers like Claude Code that need permission to copy credentials. Copies the user's auth credentials to an isolated config directory.

```json
{"provider": "claude_code"}
```

### `POST /api/providers/test`

Test a provider's connection. Returns latency and a sample response.

```json
{"provider": "claude_code", "model": "sonnet"}
```

**Response:**
```json
{
  "success": true,
  "provider": "claude_code",
  "model": "sonnet",
  "latency_ms": 5200,
  "response_preview": "Hello!",
  "error": null
}
```

---

## Collections

### `GET /api/collections`

List all indexed collections with episode counts.

**Response:**
```json
{
  "total_chunks": 1542,
  "collections": [
    {
      "collection": "blender_rigging_101",
      "collection_display": "Blender Rigging 101",
      "topic": "3d",
      "subtopic": "blender",
      "episode_count": 12,
      "episodes": [
        {"episode_num": 1, "episode_title": "Introduction"},
        {"episode_num": 2, "episode_title": "Setting Up the Mesh"}
      ]
    }
  ]
}
```

### `DELETE /api/collections`

Remove a collection and all its chunks.

```json
{"collection": "blender_rigging_101"}
```

---

## Sessions

### `GET /api/sessions`

List chat sessions, most recent first.

Query params: `limit` (default 50), `offset` (default 0).

**Response:**
```json
{
  "sessions": [
    {
      "id": "a1b2c3d4e5f6",
      "title": "How do I rig a character in Blender?",
      "created_at": "2026-03-27T10:30:00Z",
      "updated_at": "2026-03-27T10:35:00Z",
      "provider": "kilo",
      "model": "kilo-auto/free",
      "message_count": 4
    }
  ]
}
```

### `GET /api/sessions/{session_id}`

Full session with all messages and their sources.

### `PATCH /api/sessions/{session_id}`

Rename a session.

```json
{"title": "Blender Rigging Chat"}
```

### `DELETE /api/sessions/{session_id}`

Delete a session and all its messages (cascade).

### `POST /api/sessions/search`

Full-text search across all message history (uses FTS5).

```json
{"query": "weight painting", "limit": 20}
```
