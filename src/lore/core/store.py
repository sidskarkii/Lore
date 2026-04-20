"""Store — LanceDB wrapper for chunk storage, retrieval, and management."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import lancedb
import pandas as pd
import pyarrow as pa

from .chunk import fmt_timestamp
from .config import get_config
from .embed import embed_dim, embed_texts


def _deep_link_url(url: str, start_sec: int) -> str:
    """Append timestamp to YouTube/Bilibili URLs for deep-linking."""
    if not url or start_sec <= 0:
        return url
    if "youtube.com" in url or "youtu.be" in url:
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}t={start_sec}"
    if "bilibili.com" in url:
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}t={start_sec}"
    return url


def _schema(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("id",               pa.string()),
        pa.field("text",             pa.string()),
        pa.field("vector",           pa.list_(pa.float32(), dim)),
        pa.field("collection",       pa.string()),
        pa.field("collection_display", pa.string()),
        pa.field("topic",            pa.string()),
        pa.field("subtopic",         pa.string()),
        pa.field("episode_num",      pa.int32()),
        pa.field("episode_title",    pa.string()),
        pa.field("url",              pa.string()),
        pa.field("source_type",      pa.string()),
        # temporal location (video/audio)
        pa.field("start_sec",        pa.int32()),
        pa.field("end_sec",          pa.int32()),
        pa.field("timestamp",        pa.string()),
        # document location (PDF, EPUB, text)
        pa.field("page_num",         pa.int32()),
        pa.field("section_heading",  pa.string()),
        pa.field("chapter",          pa.string()),
        # code location
        pa.field("line_start",       pa.int32()),
        pa.field("line_end",         pa.int32()),
        # chunk index within episode (for document expansion)
        pa.field("chunk_index",      pa.int32()),
        # enrichment
        pa.field("title",            pa.string()),
        pa.field("summary",          pa.string()),
        pa.field("keywords",         pa.string()),
        pa.field("entities",         pa.string()),
        pa.field("questions",        pa.string()),
        pa.field("semantic_key",     pa.string()),
        pa.field("language",         pa.string()),
        pa.field("file_path",        pa.string()),
    ])


_store_instance: "Store | None" = None


def get_store() -> "Store":
    """Return the process-wide Store singleton."""
    global _store_instance
    if _store_instance is None:
        _store_instance = Store()
    return _store_instance


class Store:
    """LanceDB-backed vector store for tutorial chunks."""

    def __init__(self, db_path: str | Path | None = None):
        cfg = get_config()
        if db_path is None:
            db_path = cfg.resolve_path("store.path")
        self._db = lancedb.connect(str(db_path))
        self._table_name = cfg.get("store.table", "tutorials")
        self._dim = embed_dim()
        self._table = None  # cached table handle — opened once, reused

    def _get_or_create_table(self):
        if self._table is not None:
            return self._table
        if self._table_name in self._db.list_tables().tables:
            tbl = self._db.open_table(self._table_name)
            existing_cols = {f.name for f in tbl.schema}
            required_cols = {f.name for f in _schema(self._dim)}
            if not required_cols.issubset(existing_cols):
                self._db.drop_table(self._table_name)
                tbl = self._db.create_table(self._table_name, schema=_schema(self._dim))
            self._table = tbl
        else:
            self._table = self._db.create_table(self._table_name, schema=_schema(self._dim))
        return self._table

    def _invalidate_table_cache(self):
        """Force re-open on next access — call after schema-changing ops."""
        self._table = None

    # ── Write ─────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[dict], meta: dict) -> int:
        """Embed and store chunks with metadata.

        chunks: [{"text": str, "start_sec": float, "end_sec": float}, ...]
        meta:   {"collection", "collection_display", "topic", "subtopic",
                 "episode_num", "episode_title", "url", "source_type"}

        Returns number of chunks stored.
        """
        if not chunks:
            return 0

        tbl = self._get_or_create_table()
        vectors = embed_texts([c["text"] for c in chunks])

        rows = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            chunk_id = f"{meta['collection']}_ep{meta['episode_num']:03d}_{i:04d}"
            start_sec = int(chunk.get("start_sec", 0))
            rows.append({
                "id":                 chunk_id,
                "text":               chunk["text"],
                "vector":             vec,
                "collection":         meta["collection"],
                "collection_display": meta["collection_display"],
                "topic":              meta.get("topic", ""),
                "subtopic":           meta.get("subtopic", ""),
                "episode_num":        int(meta["episode_num"]),
                "episode_title":      meta.get("episode_title", ""),
                "url":                _deep_link_url(meta.get("url", ""), start_sec),
                "source_type":        meta.get("source_type", "video"),
                "start_sec":          start_sec,
                "end_sec":            int(chunk.get("end_sec", 0)),
                "timestamp":          fmt_timestamp(start_sec) if start_sec > 0 else "",
                "page_num":           int(chunk.get("page_num", 0)),
                "section_heading":    chunk.get("section_heading", ""),
                "chapter":            chunk.get("chapter", ""),
                "line_start":         int(chunk.get("line_start", 0)),
                "line_end":           int(chunk.get("line_end", 0)),
                "chunk_index":        i,
                "title":              chunk.get("title", ""),
                "summary":            chunk.get("summary", ""),
                "keywords":           chunk.get("keywords", ""),
                "entities":           chunk.get("entities", ""),
                "questions":          chunk.get("questions", ""),
                "semantic_key":       chunk.get("semantic_key", ""),
                "language":           chunk.get("language", ""),
                "file_path":          chunk.get("file_path", meta.get("file_path", "")),
            })

        # Remove old chunks for this episode before re-adding
        try:
            tbl.delete(
                f"collection = '{meta['collection']}' "
                f"AND episode_num = {meta['episode_num']}"
            )
        except Exception:
            pass

        tbl.add(rows)
        self._rebuild_fts(tbl)
        self._optimize(tbl)
        # Refresh table handle after optimize — it deletes old fragment files and
        # the cached reference would point to stale paths on next access
        self._table = self._db.open_table(self._table_name)
        return len(rows)

    def _rebuild_fts(self, tbl):
        try:
            tbl.create_fts_index(
                "text",
                replace=True,
                stem=True,
                lower_case=True,
                remove_stop_words=True,
                ascii_folding=True,  # fold accented chars (e.g. resume -> resume)
                with_position=True,  # enable phrase queries
            )
        except Exception:
            pass

    def _optimize(self, tbl):
        try:
            tbl.optimize(cleanup_older_than=timedelta(seconds=0))
        except Exception:
            pass

    # ── Delete ────────────────────────────────────────────────────────

    def delete_collection(self, collection: str):
        """Remove all chunks for a collection."""
        tbl = self._get_or_create_table()
        tbl.delete(f"collection = '{collection}'")
        self._optimize(tbl)

    def delete_episode(self, collection: str, episode_num: int):
        """Remove chunks for a specific episode."""
        tbl = self._get_or_create_table()
        tbl.delete(
            f"collection = '{collection}' AND episode_num = {episode_num}"
        )
        self._optimize(tbl)

    # ── Read ──────────────────────────────────────────────────────────

    def list_collections(self) -> list[dict]:
        """List all collections with episode counts."""
        tbl = self._get_or_create_table()
        _cols = ["topic", "subtopic", "collection", "collection_display",
                 "episode_num", "episode_title"]
        rows = tbl.search().limit(100_000).to_list()
        if not rows:
            return []
        df = pd.DataFrame(rows)[_cols]
        if df.empty:
            return []

        grouped = df.drop_duplicates().groupby(
            ["topic", "subtopic", "collection", "collection_display"]
        )
        result = []
        for (topic, subtopic, coll, display), group in grouped:
            episodes = sorted(
                group[["episode_num", "episode_title"]]
                .drop_duplicates()
                .to_dict("records"),
                key=lambda x: x["episode_num"],
            )
            result.append({
                "topic": topic,
                "subtopic": subtopic,
                "collection": coll,
                "collection_display": display,
                "episode_count": len(episodes),
                "episodes": episodes,
            })
        return result

    def chunk_count(self) -> int:
        """Total number of chunks in the store."""
        tbl = self._get_or_create_table()
        return tbl.count_rows()

    def get_chunk_by_id(self, chunk_id: str) -> dict | None:
        """Look up a single chunk by its ID."""
        tbl = self._get_or_create_table()
        try:
            results = tbl.search().where(f"id = '{chunk_id}'").limit(1).to_list()
            return results[0] if results else None
        except Exception:
            return None

    # ── Search primitives ─────────────────────────────────────────────

    def vector_search(
        self,
        query_vec: list[float],
        n: int = 30,
        where: str | None = None,
    ) -> list[dict]:
        """Raw vector similarity search."""
        tbl = self._get_or_create_table()
        q = tbl.search(query_vec, vector_column_name="vector").metric("cosine").limit(n)
        if where:
            q = q.where(where, prefilter=True)
        return q.to_pandas().to_dict("records")

    def fts_search(
        self,
        query: str,
        n: int = 30,
        where: str | None = None,
    ) -> list[dict]:
        """Full-text (BM25) search."""
        tbl = self._get_or_create_table()
        try:
            q = tbl.search(query, query_type="fts").limit(n)
            if where:
                q = q.where(where, prefilter=True)
            return q.to_pandas().to_dict("records")
        except Exception:
            return []

    def get_neighbors(
        self,
        collection: str,
        episode_num: int,
        start_sec: int,
        end_sec: int,
    ) -> list[dict]:
        """Get chunks in a time window for parent expansion.

        Uses LanceDB WHERE clause instead of loading the full table into
        pandas — runs in ~100us vs ~50ms+ on large tables.
        """
        tbl = self._get_or_create_table()
        where = (
            f"collection = '{collection}' "
            f"AND episode_num = {episode_num} "
            f"AND start_sec >= {start_sec} "
            f"AND start_sec <= {end_sec}"
        )
        try:
            df = tbl.search().where(where).limit(200).to_pandas()
            return df.sort_values("start_sec").to_dict("records")
        except Exception:
            # Fallback: full pandas scan (rare — only if WHERE search fails)
            try:
                df = tbl.to_pandas()
                mask = (
                    (df["collection"] == collection)
                    & (df["episode_num"] == episode_num)
                    & (df["start_sec"] >= start_sec)
                    & (df["start_sec"] <= end_sec)
                )
                return df[mask].sort_values("start_sec").to_dict("records")
            except Exception:
                return []

    def get_neighbors_by_index(
        self,
        collection: str,
        episode_num: int,
        chunk_index_start: int,
        chunk_index_end: int,
    ) -> list[dict]:
        """Get chunks by index range for document-style expansion."""
        tbl = self._get_or_create_table()
        where = (
            f"collection = '{collection}' "
            f"AND episode_num = {episode_num} "
            f"AND chunk_index >= {chunk_index_start} "
            f"AND chunk_index <= {chunk_index_end}"
        )
        try:
            df = tbl.search().where(where).limit(200).to_pandas()
            return df.sort_values("chunk_index").to_dict("records")
        except Exception:
            return []
