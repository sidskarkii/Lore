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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import get_config

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
            db_path = cfg.resolve_path("store.path").parent / "app.db"

        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript(_SCHEMA)
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


# Module-level singleton
_db: Database | None = None


def get_database() -> Database:
    """Get or create the global database singleton."""
    global _db
    if _db is None:
        _db = Database()
    return _db
