"""Lore MCP server — exposes knowledge base tools to AI agents.

Tools (12):
    intro             — deep orientation: collections, summaries, health, workflows
    search            — hybrid search (vector + BM25 + reranking)
    search_deep       — multi-hop decomposition for complex queries
    get_context       — expand around a search result or read a section
    get_toc           — browse a collection's structure
    find_related      — cross-source entity connections
    entity_index      — view/rebuild the fuzzy entity index
    reset_session     — clear fetch history after context compaction
    ingest            — auto-detect and ingest content
    ingest_status     — check ingestion progress
    rate_result       — explicit feedback on search results
    delete_collection — remove a collection
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import uuid
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP, Context
from mcp.types import ToolAnnotations
from pydantic import Field

from ..core.config import get_config
from ..core.search import get_search_engine
from ..core.database import get_database
from ..core.store import get_store
from ..providers.registry import get_registry

_ingest_jobs: dict[str, dict] = {}
_ingest_jobs_lock = threading.Lock()
_ingest_queue: asyncio.Queue | None = None
_ingest_worker_started = False

_session_lock = threading.Lock()
_sessions: dict[str, dict] = {}


def _get_session(session_id: str) -> dict:
    with _session_lock:
        if session_id not in _sessions:
            _sessions[session_id] = {"last_shown_ids": [], "fetched_texts": {}}
        return _sessions[session_id]


_default_session_id = uuid.uuid4().hex[:12]


def _build_instructions() -> str:
    """Build dynamic MCP instructions with current store stats."""
    # Dynamic state
    state = ""
    try:
        store = get_store()
        collections = store.list_collections()
        total = store.chunk_count()
        if collections:
            topics = sorted({c["topic"] for c in collections if c["topic"]})
            names = [c["collection_display"] for c in collections]
            state = (
                f"Currently indexed: {total} chunks across {len(collections)} collections "
                f"({', '.join(names[:5])}{'...' if len(names) > 5 else ''}). "
            )
            if topics:
                state += f"Topics: {', '.join(topics)}. "

            try:
                from ..core.entities import get_entity_index
                idx = get_entity_index()
                cross = idx.get_cross_source_entities()
                if cross:
                    state += f"{len(cross)} entities bridge multiple sources — use find_related to explore. "
            except Exception:
                pass

            try:
                registry = get_registry()
                if not registry.active:
                    state += "No LLM provider configured — search_deep unavailable. "
            except Exception:
                pass
        else:
            state = "No content indexed yet — use ingest to add videos, documents, or web pages. "
    except Exception:
        pass

    return (
        "Lore is a local-first RAG knowledge base. Content is organized as collections "
        "(books, videos, docs) split into searchable chunks with metadata, entities, and "
        f"concept tags. {state}"
        "\n\n"
        "Default retrieval loop:\n"
        "1. search(query) — returns compact results (~50 tokens each): scores, titles, summaries. Keep queries short and specific.\n"
        "2. Scan results. Pick promising hits by score and summary.\n"
        "3. get_context(chunk_id) — fetch full text (~500-1000 tokens) for selected chunks. Paginate with page_tokens.\n"
        "4. find_related(chunk_id) — discover what other sources say about the same entities.\n"
        "\n"
        "Other tools:\n"
        "- intro — call for full orientation: collection summaries, topics, health, workflows\n"
        "- search_deep — multi-hop decomposition for complex/comparative questions (slower, uses LLM)\n"
        "- get_toc(collection) — browse a collection's structure by section\n"
        "- entity_index — view all known entities and cross-source connections\n"
        "- reset_session — call after context compaction so fetched chunks regain full relevance\n"
        "\n"
        "Avoid: long multi-sentence queries, compact=false before scanning results, "
        "fetching many chunks at once, search_deep for simple lookups."
    )


def create_mcp_server() -> FastMCP:
    mcp = FastMCP(
        "Lore",
        instructions=_build_instructions(),
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
    )
    _register_tools(mcp)
    return mcp


def _estimate_tokens(text: str) -> int:
    return len(text) // 4 if text else 0


def _source_location(r: dict) -> dict:
    """Return source-type-specific location fields."""
    source_type = r.get("source_type", "")
    if source_type in ("video", "audio"):
        return {
            "timestamp": r.get("timestamp", ""),
            "start_sec": r.get("start_sec", 0),
            "end_sec": r.get("end_sec", 0),
        }
    elif source_type == "code":
        return {
            "file_path": r.get("file_path", ""),
            "line_start": r.get("line_start", 0),
            "line_end": r.get("line_end", 0),
        }
    return {
        "page_num": r.get("page_num", 0),
        "section_heading": r.get("section_heading", ""),
        "chapter": r.get("chapter", ""),
    }


def _format_result(r: dict) -> dict:
    """Convert a raw search result dict to the full MCP response format."""
    result = {
        "chunk_id": r.get("id", ""),
        "text": r.get("text", ""),
        "score": r.get("_score", 0.0),
        "token_count": _estimate_tokens(r.get("text", "")),
        "collection": r.get("collection", ""),
        "collection_display": r.get("collection_display", ""),
        "episode_num": r.get("episode_num", 0),
        "episode_title": r.get("episode_title", ""),
        "source_type": r.get("source_type", ""),
        "url": r.get("url", ""),
        "topic": r.get("topic", ""),
        "subtopic": r.get("subtopic", ""),
        "title": r.get("title", ""),
        "summary": r.get("summary", ""),
        "keywords": r.get("keywords", ""),
        "concept_tags": r.get("concept_tags", ""),
        "entities": r.get("entities", ""),
        "importance": r.get("importance", 3),
        "semantic_key": r.get("semantic_key", ""),
    }
    result.update(_source_location(r))
    return result


def _format_compact_result(r: dict) -> dict:
    """Metadata-only result for progressive disclosure. No full text."""
    result = {
        "chunk_id": r.get("id", ""),
        "score": r.get("_score", 0.0),
        "token_count": _estimate_tokens(r.get("text", "")),
        "collection": r.get("collection", ""),
        "collection_display": r.get("collection_display", ""),
        "episode_title": r.get("episode_title", ""),
        "source_type": r.get("source_type", ""),
        "title": r.get("title", ""),
        "summary": r.get("summary", ""),
        "keywords": r.get("keywords", ""),
        "concept_tags": r.get("concept_tags", ""),
        "importance": r.get("importance", 3),
    }
    result.update(_source_location(r))
    return result


def _load_book_summaries() -> dict[str, dict]:
    """Load book summaries from archive for intro tool."""
    cfg = get_config()
    archive_dir = cfg.archive_dir
    summaries = {}
    if not archive_dir.exists():
        return summaries
    for coll_dir in archive_dir.iterdir():
        if not coll_dir.is_dir():
            continue
        summary_file = coll_dir / "book_summary.json"
        meta_file = coll_dir / "meta.json"
        if not summary_file.exists():
            continue
        try:
            summary = json.loads(summary_file.read_text())
            meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
            summaries[coll_dir.name] = {
                "display_name": meta.get("collection_display", coll_dir.name),
                "topic": meta.get("topic", ""),
                "subtopic": meta.get("subtopic", ""),
                "source_type": meta.get("source_type", ""),
                "chunk_count": meta.get("chunk_count", 0),
                "overview": summary.get("overview", ""),
                "main_themes": summary.get("main_themes", []),
                "tags": summary.get("tags", []),
                "key_takeaways": summary.get("key_takeaways", []),
            }
        except (json.JSONDecodeError, OSError):
            continue
    return summaries


def _register_tools(mcp: FastMCP) -> None:

    # ── intro ───────────────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def intro() -> dict:
        """Deep orientation to the Lore knowledge base.

        Call this once at the start of a session to understand what's
        available, how to use it effectively, and what topics are covered.
        Returns collection summaries, usage patterns, cross-source entities,
        and workflow tips.

        WHEN TO USE: At the beginning of a research session, or when you
        want to understand the full scope of what Lore knows. After this,
        you'll know exactly which collections to search and how to navigate.
        """
        try:
            store = get_store()
            collections = store.list_collections()
            total_chunks = store.chunk_count()
            book_summaries = _load_book_summaries()

            coll_details = []
            topic_map: dict[str, list[str]] = {}
            all_tags: list[str] = []

            for c in collections:
                coll_id = c["collection"]
                detail = {
                    "collection": coll_id,
                    "display_name": c["collection_display"],
                    "topic": c["topic"],
                    "subtopic": c["subtopic"],
                    "episode_count": c["episode_count"],
                }

                bs = book_summaries.get(coll_id)
                if bs:
                    detail["overview"] = bs["overview"]
                    themes = bs["main_themes"]
                    if themes:
                        detail["themes"] = [
                            t.get("theme", t.get("title", "")) for t in themes
                            if isinstance(t, dict)
                        ][:5]
                    detail["tags"] = bs.get("tags", [])
                    detail["chunk_count"] = bs.get("chunk_count", 0)
                    all_tags.extend(bs.get("tags", []))

                coll_details.append(detail)
                topic = c["topic"] or "uncategorized"
                topic_map.setdefault(topic, []).append(c["collection_display"])

            usage = {}
            try:
                db = get_database()
                stats = db.get_interaction_stats()
                usage["total_interactions"] = stats["total_interactions"]
                usage["unique_sessions"] = stats["unique_sessions"]

                top_queries = db.get_top_queries(5)
                if top_queries:
                    usage["popular_queries"] = top_queries

                top_chunks = db.get_top_chunks(5)
                if top_chunks:
                    usage["most_accessed_chunks"] = top_chunks
            except Exception:
                pass

            cross_source = []
            try:
                from ..core.entities import get_entity_index
                idx = get_entity_index()
                for c in idx.get_cross_source_entities()[:10]:
                    cross_source.append({
                        "entity": c.canonical,
                        "type": c.entity_type,
                        "sources": sorted(c.sources),
                        "mentions": c.count,
                    })
            except Exception:
                pass

            tag_counts: dict[str, int] = {}
            for t in all_tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1
            top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:15]

            health_info = {}
            try:
                cfg = get_config()
                registry = get_registry()
                active = registry.active
                health_info = {
                    "status": "ok",
                    "embedding_model": cfg.get("embedding.model", ""),
                    "reranker_model": cfg.get("search.reranker_model", ""),
                    "active_provider": active.name if active else None,
                }
            except Exception:
                health_info = {"status": "ok"}

            return {
                "success": True,
                "overview": {
                    "total_chunks": total_chunks,
                    "total_collections": len(collections),
                    "topics": {t: len(names) for t, names in topic_map.items()},
                    "top_tags": [t for t, _ in top_tags],
                },
                "health": health_info,
                "collections": coll_details,
                "cross_source_entities": cross_source,
                "usage": usage,
                "workflows": [
                    {
                        "name": "Research a topic",
                        "steps": [
                            "search(query) — compact results show scores, titles, summaries",
                            "Scan results, pick promising hits by score and relevance",
                            "get_context(chunk_id) — fetch full text for selected chunks",
                            "find_related(chunk_id) — discover what other sources say about the same entities",
                        ],
                    },
                    {
                        "name": "Browse a book's structure",
                        "steps": [
                            "intro() — see what's indexed (or check collection IDs in the intro response)",
                            "get_toc(collection) — see chapters/sections with chunk counts",
                            "get_context(chunk_id) — read a specific section",
                        ],
                    },
                    {
                        "name": "Cross-source discovery",
                        "steps": [
                            "Check cross_source_entities above for entities bridging multiple books",
                            "find_related(entity='entity name') — find all chunks mentioning an entity",
                            "search_deep(query) — multi-hop search across sources for complex questions",
                            "entity_index() — full entity map with variants and types",
                        ],
                    },
                ],
                "tips": [
                    "Search results are compact by default — scan scores and summaries first, then get_context for full text",
                    "Use get_toc to understand a book's structure before diving into specific sections",
                    "find_related discovers cross-source connections that pure text search would miss",
                    "After context compaction, call reset_session so previously-seen chunks surface at full relevance again",
                    "Chunks auto-expire from deprioritization after 30 min — reset_session is for immediate refresh",
                ],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── search ───────────────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def search(
        query: Annotated[str, Field(description="Natural language search query. Be specific for best results.")],
        n_results: Annotated[int, Field(default=5, ge=1, le=20, description="Number of results to return.")] = 5,
        topic: Annotated[str | None, Field(default=None, description="Filter by topic (e.g. '3d', 'ai', 'code'). Use intro to see available topics.")] = None,
        subtopic: Annotated[str | None, Field(default=None, description="Filter by subtopic (e.g. 'blender', 'houdini').")] = None,
        compact: Annotated[bool, Field(default=True, description="If true (default), return metadata only (title, summary, score, token_count, location) without full text. Set false to include full chunk text inline.")] = True,
        expand: Annotated[bool, Field(default=False, description="If true, return parent-expanded context (surrounding chunks) for each result. Only applies when compact=false.")] = False,
    ) -> dict:
        """Step 1: Search the knowledge base. Returns compact results (~50 tokens each).

        WHEN TO USE: Primary tool for finding information. Use for factual
        queries, how-to questions, or locating specific content. Keep queries
        short and specific. For complex multi-topic questions, use search_deep.

        RETURNS: Compact results by default — score, title, summary, keywords,
        location. Scan these first, then call get_context for full text of
        promising hits. Set compact=false only when you need inline text.
        """
        try:
            engine = get_search_engine()
            results = engine.search(
                query=query,
                n_results=n_results,
                topic=topic,
                subtopic=subtopic,
                expand=expand and not compact,
                session_id=_default_session_id,
            )
            formatter = _format_compact_result if compact else _format_result
            formatted = [formatter(r) for r in results]

            shown_ids = [f["chunk_id"] for f in formatted]
            session = _get_session(_default_session_id)
            session["last_shown_ids"] = shown_ids
            try:
                get_database().log_interaction(
                    session_id=_default_session_id, action="search",
                    query=query, chunk_ids_shown=shown_ids,
                )
            except Exception:
                pass

            return {
                "success": True,
                "query": query,
                "total": len(results),
                "compact": compact,
                "results": formatted,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "query": query, "total": 0, "results": []}

    # ── search_deep ──────────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def search_deep(
        query: Annotated[str, Field(description="Complex search query that may span multiple topics.")],
        n_results: Annotated[int, Field(default=5, ge=1, le=20, description="Number of results to return.")] = 5,
        topic: Annotated[str | None, Field(default=None, description="Filter by topic.")] = None,
        subtopic: Annotated[str | None, Field(default=None, description="Filter by subtopic.")] = None,
        compact: Annotated[bool, Field(default=True, description="If true (default), return metadata only without full text. Set false to include full chunk text.")] = True,
    ) -> dict:
        """Deep search using multi-hop query decomposition.

        Breaks complex questions into simpler sub-queries using an LLM,
        runs hybrid search for each, then fuses and reranks all results.
        Slower than search (makes LLM calls) but better for questions that
        need information from multiple sources.

        WHEN TO USE: For complex questions like "Compare how Blender and
        Houdini handle fluid simulation" or "What are all the steps to set
        up a full render pipeline?" Falls back to regular search if no LLM
        provider is configured.

        RETURNS: Same format as search. Compact by default (metadata only).
        Set compact=false for full text inline.

        REQUIRES: An active LLM provider configured in Lore (e.g. OpenRouter
        via config.local.yaml). Without one, behaves identically to search.
        """
        try:
            engine = get_search_engine()
            registry = get_registry()
            provider = registry.active

            results = engine.search_multi_hop(
                query=query,
                provider=provider,
                n_results=n_results,
                topic=topic,
                subtopic=subtopic,
                session_id=_default_session_id,
            )
            formatter = _format_compact_result if compact else _format_result
            formatted = [formatter(r) for r in results]

            shown_ids = [f["chunk_id"] for f in formatted]
            session = _get_session(_default_session_id)
            session["last_shown_ids"] = shown_ids
            try:
                get_database().log_interaction(
                    session_id=_default_session_id, action="search_deep",
                    query=query, chunk_ids_shown=shown_ids,
                )
            except Exception:
                pass

            return {
                "success": True,
                "query": query,
                "total": len(results),
                "compact": compact,
                "results": formatted,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "query": query, "total": 0, "results": []}

    # ── get_context ──────────────────────────────────────────────────

    # Chunk IDs are now domain-specific — look up metadata from store instead of parsing

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def get_context(
        chunk_id: Annotated[str | None, Field(default=None, description="Chunk ID from a search result. Provide this OR collection+episode_num+start_sec.")] = None,
        collection: Annotated[str | None, Field(default=None, description="Collection name.")] = None,
        episode_num: Annotated[int | None, Field(default=None, description="Episode number within the collection.")] = None,
        start_sec: Annotated[int | None, Field(default=None, description="Start of the time window in seconds.")] = None,
        end_sec: Annotated[int | None, Field(default=None, description="End of the time window in seconds.")] = None,
        direction: Annotated[str, Field(default="around", description="Direction to expand: 'before', 'after', or 'around' the target.")] = "around",
        amount_sec: Annotated[int, Field(default=300, ge=30, le=1800, description="How many seconds of context to fetch.")] = 300,
        page: Annotated[int, Field(default=1, ge=1, description="Page number (1-indexed). Use with page_tokens to paginate through content.")] = 1,
        page_tokens: Annotated[int, Field(default=1500, ge=0, description="Max tokens per page. 0 = no pagination (return all). Default 1500. Agent controls how much to read per page.")] = 1500,
    ) -> dict:
        """Step 3: Fetch full text (~500-1000 tokens) around a search result.

        WHEN TO USE: After scanning compact search results, call this with
        the chunk_id of promising hits. Paginate with page_tokens to control
        how much text per response.

        Pass chunk_id from search results. Or specify collection + episode_num
        + start_sec directly. Use direction ('before'/'after'/'around') and
        page/page_tokens to navigate through content.
        """
        try:
            store = get_store()

            if chunk_id:
                chunk = store.get_chunk_by_id(chunk_id)
                if not chunk:
                    return {"success": False, "error": f"Chunk not found: {chunk_id}"}

                collection = chunk.get("collection", "")
                episode_num = int(chunk.get("episode_num", 1))
                has_timestamps = int(chunk.get("start_sec", 0)) > 0 or int(chunk.get("end_sec", 0)) > 0
            elif collection is not None and episode_num is not None and start_sec is not None:
                has_timestamps = True
            else:
                return {"success": False, "error": "Provide chunk_id OR (collection + episode_num + start_sec)"}

            if has_timestamps:
                center = int(chunk.get("start_sec", 0)) if chunk_id else start_sec
                if direction == "before":
                    window_start = max(0, center - amount_sec)
                    window_end = center
                elif direction == "after":
                    window_start = center
                    window_end = center + amount_sec
                else:
                    half = amount_sec // 2
                    window_start = max(0, center - half)
                    window_end = center + half

                neighbors = store.get_neighbors(
                    collection=collection,
                    episode_num=episode_num,
                    start_sec=window_start,
                    end_sec=window_end,
                )
            else:
                idx = int(chunk.get("chunk_index", 0)) if chunk_id else 0
                expand_n = max(2, amount_sec // 60)
                if direction == "before":
                    idx_start = max(0, idx - expand_n)
                    idx_end = idx
                elif direction == "after":
                    idx_start = idx
                    idx_end = idx + expand_n
                else:
                    idx_start = max(0, idx - expand_n)
                    idx_end = idx + expand_n

                neighbors = store.get_neighbors_by_index(
                    collection=collection,
                    episode_num=episode_num,
                    chunk_index_start=idx_start,
                    chunk_index_end=idx_end,
                )

            all_chunks = [_format_result(row) for row in neighbors]

            # Dedup: remove chunks fetched within TTL window
            import time as _time
            session = _get_session(_default_session_id)
            fetched_texts = session.get("fetched_texts", {})
            ttl_sec = get_config().get("search.session_ttl_minutes", 30) * 60
            now = _time.time()
            # Expire old entries
            expired = [k for k, t in fetched_texts.items() if now - t > ttl_sec]
            for k in expired:
                del fetched_texts[k]
            before_dedup = len(all_chunks)
            all_chunks = [c for c in all_chunks if c["chunk_id"] not in fetched_texts]
            if before_dedup > len(all_chunks):
                print(f"  [dedup] Removed {before_dedup - len(all_chunks)} already-fetched chunks")

            if page_tokens > 0 and all_chunks:
                pages: list[list[dict]] = [[]]
                current_tokens = 0
                for c in all_chunks:
                    ct = c.get("token_count", 0)
                    if current_tokens + ct > page_tokens and pages[-1]:
                        pages.append([])
                        current_tokens = 0
                    pages[-1].append(c)
                    current_tokens += ct

                total_pages = len(pages)
                page_idx = min(page, total_pages) - 1
                chunks = pages[page_idx]
            else:
                chunks = all_chunks
                total_pages = 1

            fetched_ids = [c["chunk_id"] for c in chunks]

            # Track fetched chunks for dedup (with timestamp for TTL expiry)
            for c in chunks:
                session["fetched_texts"][c["chunk_id"]] = _time.time()

            last_shown = session.get("last_shown_ids", [])
            ignored_from_last = [cid for cid in last_shown if cid not in fetched_ids] if last_shown else []
            try:
                get_database().log_interaction(
                    session_id=_default_session_id, action="get_context",
                    chunk_ids_fetched=fetched_ids,
                    chunk_ids_shown=last_shown if last_shown else None,
                    chunk_ids_ignored=ignored_from_last if ignored_from_last else None,
                )
            except Exception:
                pass

            result = {
                "success": True,
                "collection": collection,
                "episode_num": episode_num,
                "direction": direction,
                "total": len(all_chunks),
                "page": page,
                "total_pages": total_pages,
                "chunks": chunks,
            }
            if has_timestamps:
                result["window_start_sec"] = window_start
                result["window_end_sec"] = window_end
            else:
                result["chunk_index_start"] = idx_start
                result["chunk_index_end"] = idx_end
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── ingest worker (sequential queue) ──────────────────────────────

    async def _ingest_worker():
        while True:
            job_id, run_fn = await _ingest_queue.get()
            try:
                await run_fn()
            except Exception as e:
                _ingest_jobs[job_id]["status"] = "error"
                _ingest_jobs[job_id]["error"] = str(e)
            _ingest_queue.task_done()

    # ── ingest ───────────────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
    async def ingest(
        source: Annotated[str, Field(description="YouTube URL, web page URL, file path, or folder path to ingest.")],
        name: Annotated[str, Field(description="Display name for the collection (e.g. 'Blender Donut Tutorial').")],
        topic: Annotated[str, Field(default="", description="Topic category (e.g. '3d', 'ai', 'code').")] = "",
        subtopic: Annotated[str, Field(default="", description="Subtopic (e.g. 'blender', 'python').")] = "",
    ) -> dict:
        """Ingest content into the knowledge base (non-blocking).

        Queues the ingestion job and returns immediately with a job_id.
        Use ingest_status(job_id) to check progress. You can keep
        searching and working while ingestion runs in the background.

        Auto-detects the source type:
        - YouTube URL (youtube.com or youtu.be) -> downloads transcript/audio
        - Web URL (http/https) -> extracts article content
        - Directory path -> ingests all video/audio files in folder
        - File path -> ingests document (PDF, EPUB, markdown, code, audio/video)

        RETURNS: Dict with job_id to track progress via ingest_status.
        """
        from ..core.ingest import Ingester, IngestionProgress

        is_youtube = bool(re.search(r"(youtube\.com|youtu\.be)", source))
        is_url = source.startswith("http://") or source.startswith("https://")
        is_dir = Path(source).is_dir()
        is_file = Path(source).is_file()

        if not (is_youtube or is_url or is_dir or is_file):
            return {"success": False, "error": f"Source not found or unrecognized: {source}"}

        job_id = uuid.uuid4().hex[:8]
        with _ingest_jobs_lock:
            _ingest_jobs[job_id] = {
                "status": "queued",
                "source": source,
                "name": name,
                "chunks": 0,
                "message": "Queued for ingestion",
                "error": None,
            }

        async def _run_ingest():
            try:
                with _ingest_jobs_lock:
                    _ingest_jobs[job_id]["status"] = "running"
                    _ingest_jobs[job_id]["message"] = "Starting ingestion..."
                ingester = Ingester()

                def on_progress(p: IngestionProgress):
                    with _ingest_jobs_lock:
                        _ingest_jobs[job_id]["message"] = f"{p.stage}: {p.message or ''}"

                kwargs = dict(name=name, topic=topic, subtopic=subtopic, on_progress=on_progress)

                if is_youtube:
                    chunks = await asyncio.to_thread(ingester.ingest_youtube, url=source, **kwargs)
                elif is_url:
                    chunks = await asyncio.to_thread(ingester.ingest_url, url=source, **kwargs)
                elif is_dir:
                    chunks = await asyncio.to_thread(ingester.ingest_folder, folder=source, **kwargs)
                else:
                    chunks = await asyncio.to_thread(ingester.ingest_file, path=source, **kwargs)

                with _ingest_jobs_lock:
                    _ingest_jobs[job_id]["status"] = "done"
                    _ingest_jobs[job_id]["chunks"] = chunks
                    _ingest_jobs[job_id]["message"] = f"Ingested {chunks} chunks"

            except Exception as e:
                with _ingest_jobs_lock:
                    _ingest_jobs[job_id]["status"] = "error"
                    _ingest_jobs[job_id]["error"] = str(e)
                    _ingest_jobs[job_id]["message"] = f"Failed: {e}"

        global _ingest_queue, _ingest_worker_started
        if _ingest_queue is None:
            _ingest_queue = asyncio.Queue()
        if not _ingest_worker_started:
            _ingest_worker_started = True
            asyncio.create_task(_ingest_worker())

        await _ingest_queue.put((job_id, _run_ingest))

        return {
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "message": f"Ingestion queued for {source}. Use ingest_status(job_id='{job_id}') to check progress.",
        }

    # ── ingest_status ───────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def ingest_status(
        job_id: Annotated[str | None, Field(default=None, description="Job ID from ingest call. Omit to see all jobs.")] = None,
    ) -> dict:
        """Check the status of an ingestion job.

        WHEN TO USE: After calling ingest, to check if it's still running,
        completed, or failed. Omit job_id to see all active/recent jobs.

        RETURNS: Job status (queued, running, done, error), progress message,
        and chunk count when complete.
        """
        if job_id:
            with _ingest_jobs_lock:
                job = _ingest_jobs.get(job_id)
                if not job:
                    return {"success": False, "error": f"Unknown job: {job_id}"}
                return {"success": True, "job_id": job_id, **dict(job)}

        with _ingest_jobs_lock:
            return {
                "success": True,
                "jobs": {jid: dict(j) for jid, j in _ingest_jobs.items()},
            }

    # ── rate_result ──────────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def rate_result(
        chunk_id: Annotated[str, Field(description="Chunk ID to rate.")],
        useful: Annotated[bool, Field(description="True if the chunk was useful, false if not.")],
    ) -> dict:
        """Rate a search result as useful or not useful.

        WHEN TO USE: After reading a chunk's full text, tell Lore whether
        it was helpful. This improves future search rankings over time.
        Explicit ratings count as strong signal.
        """
        try:
            db = get_database()
            db.rate_chunk(chunk_id, useful)
            db.log_interaction(
                session_id=_default_session_id, action="rate",
                chunk_ids_rated=[chunk_id],
                rating=1 if useful else -1,
            )
            return {"success": True, "chunk_id": chunk_id, "rated": "useful" if useful else "not useful"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── reset_session ───────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def reset_session() -> dict:
        """Reset the session's fetch history so all chunks are full-score again.

        WHEN TO USE: After you've compacted your context and no longer have
        previous search results in memory. This clears the deprioritization
        of previously-fetched chunks so they can surface at full relevance
        again. Also useful at the start of a new research direction within
        the same session.

        NOTE: Chunks also automatically become full-score again after 30
        minutes (configurable TTL), so this is only needed for immediate
        reset.
        """
        try:
            db = get_database()
            db.reset_session_fetched(_default_session_id)
            with _session_lock:
                if _default_session_id in _sessions:
                    _sessions[_default_session_id]["last_shown_ids"] = []
                    _sessions[_default_session_id]["fetched_texts"] = {}
            return {"success": True, "message": "Session fetch history cleared. All chunks are full-score eligible."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── get_toc ──────────────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def get_toc(
        collection: Annotated[str, Field(description="Collection ID (from intro or search results).")],
    ) -> dict:
        """Get the table of contents for a collection.

        Returns the document structure: sections/chapters in reading order,
        each with chunk count, token estimate, and first_chunk_id for
        navigating directly to that section via get_context.

        WHEN TO USE: To understand what a book/document covers before
        searching. Lets you browse by structure instead of keyword.
        """
        try:
            store = get_store()
            sections = store.get_toc(collection)
            total_tokens = sum(s["token_count"] for s in sections)
            return {
                "success": True,
                "collection": collection,
                "total_sections": len(sections),
                "total_tokens": total_tokens,
                "sections": sections,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── delete_collection ────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
    def delete_collection(
        collection: Annotated[str, Field(description="Collection ID to delete (from intro or search results).")],
    ) -> dict:
        """Permanently delete a collection and all its chunks.

        WARNING: This cannot be undone. All indexed content for the collection
        will be removed. The agent should confirm with the user before calling.

        WHEN TO USE: When the user explicitly asks to remove indexed content.
        """
        try:
            store = get_store()
            store.delete_collection(collection)
            return {"success": True, "deleted": collection}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── find_related ──────────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def find_related(
        chunk_id: Annotated[str | None, Field(default=None, description="Find chunks related to this chunk via shared entities.")] = None,
        entity: Annotated[str | None, Field(default=None, description="Find chunks mentioning this entity (fuzzy-matched).")] = None,
        collection: Annotated[str | None, Field(default=None, description="Limit results to this collection.")] = None,
        n_results: Annotated[int, Field(default=10, ge=1, le=50, description="Number of results.")] = 10,
    ) -> dict:
        """Step 4: Discover cross-source connections via shared entities.

        WHEN TO USE: After finding a useful chunk, discover what other
        sources say about the same people, concepts, or organizations.
        Resolves name variants automatically (fuzzy matching).

        Provide chunk_id (finds chunks sharing its entities) or entity
        name (finds all chunks mentioning that entity across collections).
        """
        from ..core.entities import get_entity_index

        try:
            idx = get_entity_index()
            store = get_store()

            target_entities: list[str] = []

            if chunk_id:
                chunk = store.get_chunk_by_id(chunk_id)
                if not chunk:
                    return {"success": False, "error": f"Chunk not found: {chunk_id}"}
                ents_raw = chunk.get("entities", "")
                if ents_raw:
                    try:
                        ents = json.loads(ents_raw) if isinstance(ents_raw, str) else ents_raw
                        target_entities = [e.get("name", "") for e in ents if isinstance(e, dict) and e.get("name")]
                    except (json.JSONDecodeError, TypeError):
                        pass
                if not target_entities:
                    return {"success": True, "chunk_id": chunk_id, "message": "No entities found on this chunk", "results": []}

            elif entity:
                target_entities = [entity]
            else:
                return {"success": False, "error": "Provide chunk_id or entity name"}

            canonical_entities = set()
            for ent in target_entities:
                cluster = idx.resolve(ent)
                if cluster:
                    canonical_entities.add(cluster.canonical)

            if not canonical_entities:
                canonical_entities = {e.lower() for e in target_entities}

            all_variant_names = set()
            for canonical in canonical_entities:
                for cluster in idx.clusters:
                    if cluster.canonical == canonical:
                        all_variant_names.update(v.lower() for v in cluster.variants)

            if not all_variant_names:
                all_variant_names = {e.lower() for e in target_entities}

            collections = store.list_collections()
            related_chunks = []

            for coll in collections:
                if collection and coll["collection"] != collection:
                    continue
                try:
                    chunks = store.get_all_chunks(coll["collection"])
                except Exception:
                    continue

                for c in chunks:
                    if chunk_id and c.get("id") == chunk_id:
                        continue
                    ents_raw = c.get("entities", "")
                    if not ents_raw:
                        continue
                    try:
                        ents = json.loads(ents_raw) if isinstance(ents_raw, str) else ents_raw
                        chunk_ent_names = [e.get("name", "") for e in ents if isinstance(e, dict) and e.get("name")]
                    except (json.JSONDecodeError, TypeError):
                        continue

                    shared = set()
                    for chunk_ent in chunk_ent_names:
                        cluster = idx.resolve(chunk_ent)
                        if cluster and cluster.canonical in canonical_entities:
                            shared.add(cluster.canonical)

                    if shared:
                        related_chunks.append({
                            "chunk_id": c.get("id", ""),
                            "collection": c.get("collection", ""),
                            "episode_title": c.get("episode_title", ""),
                            "shared_entities": sorted(shared),
                            "match_count": len(shared),
                        })

            related_chunks.sort(key=lambda x: x["match_count"], reverse=True)
            total_found = len(related_chunks)
            related_chunks = related_chunks[:n_results]

            return {
                "success": True,
                "query_entities": sorted(canonical_entities),
                "total_related": total_found,
                "returned": len(related_chunks),
                "results": related_chunks,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── entity_index ────────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def entity_index(
        rebuild: Annotated[bool, Field(default=False, description="Force rebuild the entity index from all chunks.")] = False,
    ) -> dict:
        """View or rebuild the fuzzy entity index.

        Shows all canonical entities with their variants, types, and which
        collections they appear in. Identifies cross-source entities that
        bridge multiple books/documents.

        WHEN TO USE: To understand what entities exist across your knowledge
        base, find cross-source connections, or rebuild after new ingestion.
        """
        from ..core.entities import get_entity_index

        try:
            idx = get_entity_index(rebuild=rebuild)
            stats = idx.stats()
            cross = idx.get_cross_source_entities()
            stats["cross_source_details"] = [
                {
                    "canonical": c.canonical,
                    "type": c.entity_type,
                    "sources": sorted(c.sources),
                    "variants": sorted(c.variants),
                    "count": c.count,
                }
                for c in sorted(cross, key=lambda x: x.count, reverse=True)[:20]
            ]
            return {"success": True, **stats}
        except Exception as e:
            return {"success": False, "error": str(e)}

