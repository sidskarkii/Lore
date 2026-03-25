"""Store — LanceDB wrapper for chunk storage, retrieval, and management."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pyarrow as pa

from .chunk import fmt_timestamp
from .config import get_config
from .embed import embed_dim, embed_texts


def _schema(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("id",               pa.string()),
        pa.field("text",             pa.string()),
        pa.field("vector",           pa.list_(pa.float32(), dim)),
        pa.field("collection",       pa.string()),   # replaces "tutorial"
        pa.field("collection_display", pa.string()),
        pa.field("topic",            pa.string()),
        pa.field("subtopic",         pa.string()),
        pa.field("episode_num",      pa.int32()),
        pa.field("episode_title",    pa.string()),
        pa.field("url",              pa.string()),
        pa.field("source_type",      pa.string()),   # "video", "docs", "qa"
        pa.field("start_sec",        pa.int32()),
        pa.field("end_sec",          pa.int32()),
        pa.field("timestamp",        pa.string()),
    ])


class Store:
    """LanceDB-backed vector store for tutorial chunks."""

    def __init__(self, db_path: str | Path | None = None):
        import lancedb

        cfg = get_config()
        if db_path is None:
            db_path = cfg.resolve_path("store.path")
        self._db = lancedb.connect(str(db_path))
        self._table_name = cfg.get("store.table", "tutorials")
        self._dim = embed_dim()

    def _get_or_create_table(self):
        if self._table_name in self._db.table_names():
            return self._db.open_table(self._table_name)
        return self._db.create_table(self._table_name, schema=_schema(self._dim))

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
                "url":                meta.get("url", ""),
                "source_type":        meta.get("source_type", "video"),
                "start_sec":          int(chunk["start_sec"]),
                "end_sec":            int(chunk["end_sec"]),
                "timestamp":          fmt_timestamp(chunk["start_sec"]),
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
        return len(rows)

    def _rebuild_fts(self, tbl):
        try:
            tbl.create_fts_index("text", replace=True)
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
        df = tbl.to_pandas()[
            ["topic", "subtopic", "collection", "collection_display",
             "episode_num", "episode_title"]
        ]
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
        """Get chunks in a time window for parent expansion."""
        tbl = self._get_or_create_table()
        df = tbl.to_pandas()
        mask = (
            (df["collection"] == collection)
            & (df["episode_num"] == episode_num)
            & (df["start_sec"] >= start_sec)
            & (df["start_sec"] <= end_sec)
        )
        return df[mask].sort_values("start_sec").to_dict("records")
