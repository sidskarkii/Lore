"""Database — SQLite persistence for chat sessions, messages, and settings.

Single file at data/app.db (configurable). Auto-creates tables on first use.
Uses stdlib sqlite3 — zero extra dependencies.

Tables:
    chat_sessions  — conversation sessions with title, provider, model
    messages       — individual messages with role, content, sources
    messages_fts   — FTS5 index for searching message history
    settings       — key-value store for app state and preferences
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import get_config

_INTERACTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('search', 'search_deep', 'get_context', 'get_toc', 'ingest', 'rate')),
    query TEXT,
    chunk_ids_shown TEXT,
    chunk_ids_fetched TEXT,
    chunk_ids_ignored TEXT,
    chunk_ids_rated TEXT,
    rating INTEGER,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_interactions_session
    ON interactions(session_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_interactions_action
    ON interactions(action, timestamp);

CREATE TABLE IF NOT EXISTS chunk_ratings (
    chunk_id TEXT PRIMARY KEY,
    fetches INTEGER NOT NULL DEFAULT 0,
    ignores INTEGER NOT NULL DEFAULT 0,
    explicit_up INTEGER NOT NULL DEFAULT 0,
    explicit_down INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    collection TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('started', 'enriching', 'storing', 'done', 'failed')),
    chunks INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    error TEXT
);
"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    provider TEXT,
    model TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    sources_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

# FTS5 is a separate statement — might fail if SQLite was compiled without it
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
    USING fts5(content, content=messages, content_rowid=rowid);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES('delete', old.rowid, old.content);
END;
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class Database:
    """SQLite database for chat persistence and app settings."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            cfg = get_config()
            db_path = cfg.data_dir / "app.db"

        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript(_SCHEMA)
        self._conn.executescript(_INTERACTION_SCHEMA)
        try:
            self._conn.executescript(_FTS_SCHEMA)
        except sqlite3.OperationalError:
            pass  # FTS5 not available — search will fall back to LIKE
        self._conn.commit()

    def close(self):
        self._conn.close()

    # ── Sessions ──────────────────────────────────────────────────────

    def create_session(
        self,
        title: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict:
        """Create a new chat session. Returns the session dict."""
        now = _now()
        session_id = _new_id()
        self._conn.execute(
            "INSERT INTO chat_sessions (id, title, created_at, updated_at, provider, model) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, title or "New Chat", now, now, provider, model),
        )
        self._conn.commit()
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> dict | None:
        """Get a session by ID, including its messages."""
        row = self._conn.execute(
            "SELECT * FROM chat_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None

        messages = self._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()

        return {
            **dict(row),
            "messages": [
                {
                    "id": m["id"],
                    "role": m["role"],
                    "content": m["content"],
                    "sources": json.loads(m["sources_json"]) if m["sources_json"] else [],
                    "created_at": m["created_at"],
                }
                for m in messages
            ],
        }

    def list_sessions(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """List sessions, most recent first. Includes message count, not messages."""
        rows = self._conn.execute(
            "SELECT s.*, COUNT(m.id) as message_count "
            "FROM chat_sessions s "
            "LEFT JOIN messages m ON m.session_id = s.id "
            "GROUP BY s.id "
            "ORDER BY s.updated_at DESC "
            "LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_session_title(self, session_id: str, title: str):
        """Rename a session."""
        self._conn.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), session_id),
        )
        self._conn.commit()

    def delete_session(self, session_id: str):
        """Delete a session and all its messages."""
        self._conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        self._conn.commit()

    # ── Messages ──────────────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list[dict] | None = None,
    ) -> dict:
        """Add a message to a session. Returns the message dict."""
        now = _now()
        msg_id = _new_id()
        sources_json = json.dumps(sources) if sources else None

        self._conn.execute(
            "INSERT INTO messages (id, session_id, role, content, sources_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (msg_id, session_id, role, content, sources_json, now),
        )
        # Update session timestamp
        self._conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        self._conn.commit()

        return {
            "id": msg_id,
            "role": role,
            "content": content,
            "sources": sources or [],
            "created_at": now,
        }

    # ── Search ────────────────────────────────────────────────────────

    def search_messages(self, query: str, limit: int = 20) -> list[dict]:
        """Search message content using FTS5 (falls back to LIKE)."""
        try:
            rows = self._conn.execute(
                "SELECT m.*, s.title as session_title "
                "FROM messages_fts fts "
                "JOIN messages m ON m.rowid = fts.rowid "
                "JOIN chat_sessions s ON s.id = m.session_id "
                "WHERE messages_fts MATCH ? "
                "ORDER BY rank "
                "LIMIT ?",
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # FTS5 not available, fall back to LIKE
            rows = self._conn.execute(
                "SELECT m.*, s.title as session_title "
                "FROM messages m "
                "JOIN chat_sessions s ON s.id = m.session_id "
                "WHERE m.content LIKE ? "
                "ORDER BY m.created_at DESC "
                "LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()

        return [dict(r) for r in rows]

    # ── Settings ──────────────────────────────────────────────────────

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value by key."""
        row = self._conn.execute(
            "SELECT value_json FROM settings WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return default
        return json.loads(row["value_json"])

    def set_setting(self, key: str, value: Any):
        """Set a setting value (upsert)."""
        self._conn.execute(
            "INSERT INTO settings (key, value_json, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value_json = ?, updated_at = ?",
            (key, json.dumps(value), _now(), json.dumps(value), _now()),
        )
        self._conn.commit()


    # ── Interaction Logging ────────────────────────────────────────────

    def log_interaction(
        self,
        session_id: str,
        action: str,
        query: str | None = None,
        chunk_ids_shown: list[str] | None = None,
        chunk_ids_fetched: list[str] | None = None,
        chunk_ids_ignored: list[str] | None = None,
        chunk_ids_rated: list[str] | None = None,
        rating: int | None = None,
        metadata: dict | None = None,
    ):
        self._conn.execute(
            "INSERT INTO interactions (session_id, timestamp, action, query, "
            "chunk_ids_shown, chunk_ids_fetched, chunk_ids_ignored, chunk_ids_rated, "
            "rating, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id, _now(), action, query,
                json.dumps(chunk_ids_shown) if chunk_ids_shown else None,
                json.dumps(chunk_ids_fetched) if chunk_ids_fetched else None,
                json.dumps(chunk_ids_ignored) if chunk_ids_ignored else None,
                json.dumps(chunk_ids_rated) if chunk_ids_rated else None,
                rating,
                json.dumps(metadata) if metadata else None,
            ),
        )
        self._conn.commit()

        if chunk_ids_fetched:
            self._update_chunk_ratings(chunk_ids_fetched, "fetch")
        if chunk_ids_shown and chunk_ids_fetched:
            ignored = set(chunk_ids_shown) - set(chunk_ids_fetched)
            if ignored:
                self._update_chunk_ratings(list(ignored), "ignore")

    def _update_chunk_ratings(self, chunk_ids: list[str], signal: str):
        now = _now()
        for cid in chunk_ids:
            if signal == "fetch":
                self._conn.execute(
                    "INSERT INTO chunk_ratings (chunk_id, fetches, ignores, updated_at) "
                    "VALUES (?, 1, 0, ?) "
                    "ON CONFLICT(chunk_id) DO UPDATE SET fetches = fetches + 1, updated_at = ?",
                    (cid, now, now),
                )
            elif signal == "ignore":
                self._conn.execute(
                    "INSERT INTO chunk_ratings (chunk_id, fetches, ignores, updated_at) "
                    "VALUES (?, 0, 1, ?) "
                    "ON CONFLICT(chunk_id) DO UPDATE SET ignores = ignores + 1, updated_at = ?",
                    (cid, now, now),
                )
        self._conn.commit()

    def rate_chunk(self, chunk_id: str, useful: bool):
        now = _now()
        col = "explicit_up" if useful else "explicit_down"
        self._conn.execute(
            f"INSERT INTO chunk_ratings (chunk_id, fetches, ignores, {col}, updated_at) "
            f"VALUES (?, 0, 0, 1, ?) "
            f"ON CONFLICT(chunk_id) DO UPDATE SET {col} = {col} + 1, updated_at = ?",
            (chunk_id, now, now),
        )
        self._conn.commit()

    def get_chunk_rating(self, chunk_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM chunk_ratings WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_chunk_ratings_batch(self, chunk_ids: list[str]) -> dict[str, dict]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self._conn.execute(
            f"SELECT * FROM chunk_ratings WHERE chunk_id IN ({placeholders})", chunk_ids
        ).fetchall()
        return {row["chunk_id"]: dict(row) for row in rows}

    def get_session_fetched_ids(self, session_id: str, ttl_minutes: int = 0) -> set[str]:
        """Get chunk IDs fetched via get_context in a session.

        If ttl_minutes > 0, only return IDs fetched within the last N minutes
        (expired fetches become full-score eligible again).
        """
        if ttl_minutes > 0:
            cutoff = (datetime.now(timezone.utc) - timedelta(minutes=ttl_minutes)).isoformat()
            rows = self._conn.execute(
                "SELECT chunk_ids_fetched FROM interactions "
                "WHERE session_id = ? AND action = 'get_context' AND timestamp > ?",
                (session_id, cutoff),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT chunk_ids_fetched FROM interactions "
                "WHERE session_id = ? AND action = 'get_context'",
                (session_id,),
            ).fetchall()
        result = set()
        for row in rows:
            if row["chunk_ids_fetched"]:
                try:
                    result.update(json.loads(row["chunk_ids_fetched"]))
                except (json.JSONDecodeError, TypeError):
                    pass
        return result

    def reset_session_fetched(self, session_id: str):
        """Clear fetch history for a session (e.g. after agent context compaction)."""
        self._conn.execute(
            "DELETE FROM interactions WHERE session_id = ? AND action = 'get_context'",
            (session_id,),
        )
        self._conn.commit()

    def get_interaction_stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        sessions = self._conn.execute("SELECT COUNT(DISTINCT session_id) FROM interactions").fetchone()[0]
        rated_chunks = self._conn.execute("SELECT COUNT(*) FROM chunk_ratings").fetchone()[0]
        return {"total_interactions": total, "unique_sessions": sessions, "rated_chunks": rated_chunks}

    def get_top_queries(self, limit: int = 10) -> list[dict]:
        """Return most frequent search queries."""
        rows = self._conn.execute(
            "SELECT query, COUNT(*) as count FROM interactions "
            "WHERE action IN ('search', 'search_deep') AND query IS NOT NULL "
            "GROUP BY query ORDER BY count DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"query": r["query"], "count": r["count"]} for r in rows]

    def get_top_chunks(self, limit: int = 10) -> list[dict]:
        """Return highest-rated chunks by net score (fetches/upvotes minus ignores/downvotes)."""
        rows = self._conn.execute(
            "SELECT chunk_id, fetches, ignores, explicit_up, explicit_down "
            "FROM chunk_ratings "
            "ORDER BY (fetches + explicit_up * 3) - (ignores + explicit_down * 3) DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Ingestion Log -------------------------------------------------

    def log_ingest_start(self, collection: str, source_path: str):
        self._conn.execute(
            "INSERT INTO ingestion_log (collection, source_path, status, started_at) "
            "VALUES (?, ?, 'started', ?) "
            "ON CONFLICT(collection) DO UPDATE SET status='started', source_path=?, started_at=?, error=NULL",
            (collection, source_path, _now(), source_path, _now()),
        )
        self._conn.commit()

    def log_ingest_status(self, collection: str, status: str, chunks: int = 0, error: str | None = None):
        completed = _now() if status in ("done", "failed") else None
        self._conn.execute(
            "UPDATE ingestion_log SET status=?, chunks=?, completed_at=?, error=? WHERE collection=?",
            (status, chunks, completed, error, collection),
        )
        self._conn.commit()

    def get_ingest_log(self, collection: str | None = None) -> list[dict]:
        if collection:
            rows = self._conn.execute(
                "SELECT * FROM ingestion_log WHERE collection=?", (collection,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM ingestion_log ORDER BY started_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_resumable_ingests(self) -> list[dict]:
        """Return ingests that started but didn't complete (crashed mid-pipeline)."""
        rows = self._conn.execute(
            "SELECT * FROM ingestion_log WHERE status NOT IN ('done', 'failed')"
        ).fetchall()
        return [dict(r) for r in rows]


# Module-level singleton
_db: Database | None = None


def get_database() -> Database:
    """Get or create the global database singleton."""
    global _db
    if _db is None:
        _db = Database()
    return _db
