"""Wiki search index — LanceDB table for wiki page fragments.

Indexes wiki pages as searchable fragments (summary, claims, sections)
in a separate `wiki_pages` table. Supports vector + FTS hybrid search
with reranking, same approach as the chunk store.
"""

from __future__ import annotations

import re

import lancedb
import numpy as np
import pyarrow as pa

from .config import get_config
from .embed import embed_texts, embed_dim


_WIKI_TABLE = "wiki_pages"


def _wiki_schema(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("fragment_id", pa.utf8()),
        pa.field("page_id", pa.utf8()),
        pa.field("page_type", pa.utf8()),
        pa.field("title", pa.utf8()),
        pa.field("fragment_type", pa.utf8()),
        pa.field("text", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
        pa.field("source_collections", pa.utf8()),
        pa.field("supporting_source_count", pa.int32()),
        pa.field("corroboration_level", pa.utf8()),
        pa.field("confidence", pa.utf8()),
        pa.field("claim_count", pa.int32()),
    ])


def _get_db():
    cfg = get_config()
    db_path = cfg.resolve_path("store.path")
    return lancedb.connect(str(db_path))


def _get_or_create_table():
    db = _get_db()
    dim = embed_dim()
    if _WIKI_TABLE in db.list_tables().tables:
        return db.open_table(_WIKI_TABLE)
    return db.create_table(_WIKI_TABLE, schema=_wiki_schema(dim))


def _extract_fragments(page) -> list[dict]:
    """Split a wiki page into indexable fragments."""
    fragments = []
    page_meta = {
        "page_id": page.page_id,
        "page_type": page.page_type,
        "title": page.title,
        "source_collections": ",".join(page.source_collections),
        "supporting_source_count": page.supporting_source_count,
        "corroboration_level": page.corroboration_level,
        "confidence": page.confidence,
        "claim_count": page.generation.get("claim_count", 0) if page.generation else 0,
    }

    _SECTION_PATTERNS = [
        ("summary",      r'## (?:Summary|Overview)\s*\n(.*?)(?=\n## |\Z)'),
        ("claims",       r'## Key Claims\s*\n(.*?)(?=\n## |\Z)'),
        ("synthesis",    r'## (?:Cross-Source Synthesis|Synthesis)\s*\n(.*?)(?=\n## |\Z)'),
        ("themes",       r'## (?:Main Themes|Key Takeaways)\s*\n(.*?)(?=\n## |\Z)'),
        ("sections",     r'## (?:Sections|Cross-Section Patterns)\s*\n(.*?)(?=\n## |\Z)'),
        ("agreements",   r'## Agreements\s*\n(.*?)(?=\n## |\Z)'),
        ("tensions",     r'## Tensions & Disagreements\s*\n(.*?)(?=\n## |\Z)'),
        ("positions",    r'## Per-Source Positions\s*\n(.*?)(?=\n## |\Z)'),
    ]

    for frag_type, pattern in _SECTION_PATTERNS:
        match = re.search(pattern, page.content, re.DOTALL)
        if match:
            text = match.group(1).strip()
            if text:
                fragments.append({
                    "fragment_id": f"{page.page_id}::{frag_type}",
                    "fragment_type": frag_type,
                    "text": f"{page.title} — {frag_type}: {text}",
                    **page_meta,
                })

    if not fragments:
        fragments.append({
            "fragment_id": f"{page.page_id}::full",
            "fragment_type": "full",
            "text": f"{page.title}: {page.content[:2000]}",
            **page_meta,
        })

    return fragments


def rebuild_fts():
    """Rebuild the FTS index on the existing wiki table."""
    try:
        tbl = _get_or_create_table()
        if tbl.count_rows() == 0:
            return
        tbl.create_fts_index(
            "text",
            replace=True,
            stem=True,
            lower_case=True,
            remove_stop_words=True,
        )
    except Exception:
        pass


def index_page(page) -> int:
    """Index a single wiki page's fragments into LanceDB."""
    tbl = _get_or_create_table()

    try:
        tbl.delete(f"page_id = '{page.page_id.replace(chr(39), '')}'")
    except Exception:
        pass

    fragments = _extract_fragments(page)
    if not fragments:
        return 0

    texts = [f["text"] for f in fragments]
    vectors = embed_texts(texts)

    rows = []
    for frag, vec in zip(fragments, vectors):
        frag["vector"] = vec
        rows.append(frag)

    tbl.add(rows)
    rebuild_fts()
    return len(rows)


def remove_page(page_id: str):
    """Remove all fragments for a page from the index."""
    try:
        tbl = _get_or_create_table()
        tbl.delete(f"page_id = '{page_id.replace(chr(39), '')}'")
    except Exception:
        pass


def rebuild_index() -> int:
    """Rebuild the entire wiki search index from all wiki pages."""
    from .wiki import get_wiki_manager

    wm = get_wiki_manager()
    pages = wm.list_pages()

    db = _get_db()
    if _WIKI_TABLE in db.list_tables().tables:
        db.drop_table(_WIKI_TABLE)
    tbl = db.create_table(_WIKI_TABLE, schema=_wiki_schema(embed_dim()))

    all_fragments = []
    for meta in pages:
        page = wm.get_page(meta["page_id"])
        if not page or not page.content:
            continue
        all_fragments.extend(_extract_fragments(page))

    if not all_fragments:
        return 0

    texts = [f["text"] for f in all_fragments]
    vectors = embed_texts(texts)

    rows = []
    for frag, vec in zip(all_fragments, vectors):
        frag["vector"] = vec
        rows.append(frag)

    tbl.add(rows)
    rebuild_fts()

    print(f"  [wiki_index] Indexed {len(rows)} fragments from {len(pages)} pages")
    return len(rows)


def search_wiki(
    query: str,
    page_type: str | None = None,
    n_results: int = 8,
    include_stale: bool = False,
) -> list[dict]:
    """Search wiki pages by vector similarity + optional FTS."""
    try:
        tbl = _get_or_create_table()
        if tbl.count_rows() == 0:
            from .wiki import get_wiki_manager
            if get_wiki_manager().list_pages():
                rebuild_index()
                tbl = _get_or_create_table()
                if tbl.count_rows() == 0:
                    return []
            else:
                return []
    except Exception:
        return []

    query_vec = embed_texts([query])[0]

    where_clauses = []
    if page_type:
        where_clauses.append(f"page_type = '{page_type.replace(chr(39), '')}'")

    try:
        builder = tbl.search(query_vec).limit(n_results * 3)
        if where_clauses:
            builder = builder.where(" AND ".join(where_clauses))
        vector_results = builder.to_list()
    except Exception:
        vector_results = []

    try:
        fts_builder = tbl.search(query, query_type="fts").limit(n_results * 2)
        if where_clauses:
            fts_builder = fts_builder.where(" AND ".join(where_clauses))
        fts_results = fts_builder.to_list()
    except Exception:
        fts_results = []

    seen: dict[str, dict] = {}
    k = 60

    for rank, r in enumerate(vector_results):
        fid = r.get("fragment_id", "")
        rrf = 1.0 / (k + rank + 1)
        if fid in seen:
            seen[fid]["score"] += rrf
        else:
            seen[fid] = {
                "fragment_id": fid,
                "page_id": r.get("page_id", ""),
                "page_type": r.get("page_type", ""),
                "title": r.get("title", ""),
                "fragment_type": r.get("fragment_type", ""),
                "text": r.get("text", "")[:500],
                "source_collections": r.get("source_collections", ""),
                "supporting_source_count": r.get("supporting_source_count", 0),
                "corroboration_level": r.get("corroboration_level", ""),
                "confidence": r.get("confidence", ""),
                "score": rrf,
                "result_type": "wiki",
            }

    for rank, r in enumerate(fts_results):
        fid = r.get("fragment_id", "")
        rrf = 1.0 / (k + rank + 1)
        if fid in seen:
            seen[fid]["score"] += rrf
        else:
            seen[fid] = {
                "fragment_id": fid,
                "page_id": r.get("page_id", ""),
                "page_type": r.get("page_type", ""),
                "title": r.get("title", ""),
                "fragment_type": r.get("fragment_type", ""),
                "text": r.get("text", "")[:500],
                "source_collections": r.get("source_collections", ""),
                "supporting_source_count": r.get("supporting_source_count", 0),
                "corroboration_level": r.get("corroboration_level", ""),
                "confidence": r.get("confidence", ""),
                "score": rrf,
                "result_type": "wiki",
            }

    try:
        from .query_intent import detect_query_intent
        intent = detect_query_intent(query)
        if intent.page_type_boosts:
            for entry in seen.values():
                boost = intent.page_type_boosts.get(entry.get("page_type", ""), 0)
                if boost:
                    entry["score"] += boost
    except Exception:
        pass

    results = sorted(seen.values(), key=lambda x: -x["score"])

    if not include_stale:
        from .wiki import get_wiki_manager
        wm = get_wiki_manager()
        stale_ids = {p["page_id"] for p in wm.get_stale_pages()}
        results = [r for r in results if r["page_id"] not in stale_ids]

    unique_pages: dict[str, dict] = {}
    for r in results:
        pid = r["page_id"]
        if pid not in unique_pages or r["score"] > unique_pages[pid]["score"]:
            unique_pages[pid] = r

    return sorted(unique_pages.values(), key=lambda x: -x["score"])[:n_results]
