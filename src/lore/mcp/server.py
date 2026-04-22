"""Lore MCP server — exposes knowledge base tools to AI agents.

Tools:
    search          — hybrid search (vector + BM25 + reranking)
    search_deep     — multi-hop decomposition for complex queries
    get_context     — expand around a search result or read a section
    ingest          — auto-detect and ingest content (YouTube, file, URL, text)
    list_collections — browse indexed collections
    delete_collection — remove a collection
    health          — server status and stats
"""

from __future__ import annotations

import asyncio
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
            _sessions[session_id] = {"last_shown_ids": []}
        return _sessions[session_id]


_default_session_id = uuid.uuid4().hex[:12]


def _build_instructions() -> str:
    """Build dynamic MCP instructions with current store stats."""
    base = (
        "Lore is a local-first RAG knowledge base. It indexes videos, documents, "
        "code, and web pages into searchable chunks with timestamp-level precision."
    )
    try:
        store = get_store()
        collections = store.list_collections()
        total = store.chunk_count()
        if collections:
            topics = sorted({c["topic"] for c in collections if c["topic"]})
            names = [c["collection_display"] for c in collections]
            base += (
                f" Currently indexed: {total} chunks across {len(collections)} collections "
                f"({', '.join(names[:5])}{'...' if len(names) > 5 else ''})."
                f" Topics: {', '.join(topics)}." if topics else ""
            )
        else:
            base += " No content indexed yet — use 'ingest' to add content."
    except Exception:
        pass

    base += (
        " Use 'search' for most queries — results are compact by default (metadata "
        "only, no full text). Scan scores and summaries, then call 'get_context' to "
        "fetch full text for results you need. Use 'get_toc' to browse a collection's "
        "structure. Use 'search_deep' for complex multi-topic questions."
    )
    return base


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


def _register_tools(mcp: FastMCP) -> None:

    # ── search ───────────────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def search(
        query: Annotated[str, Field(description="Natural language search query. Be specific for best results.")],
        n_results: Annotated[int, Field(default=5, ge=1, le=20, description="Number of results to return.")] = 5,
        topic: Annotated[str | None, Field(default=None, description="Filter by topic (e.g. '3d', 'ai', 'code'). Use list_collections to see available topics.")] = None,
        subtopic: Annotated[str | None, Field(default=None, description="Filter by subtopic (e.g. 'blender', 'houdini').")] = None,
        compact: Annotated[bool, Field(default=True, description="If true (default), return metadata only (title, summary, score, token_count, location) without full text. Set false to include full chunk text inline.")] = True,
        expand: Annotated[bool, Field(default=False, description="If true, return parent-expanded context (surrounding chunks) for each result. Only applies when compact=false.")] = False,
    ) -> dict:
        """Search the Lore knowledge base using hybrid retrieval.

        Combines vector similarity search with BM25 keyword matching and
        cross-encoder reranking for high-quality results.

        WHEN TO USE: Primary tool for finding information. Use for factual
        queries, how-to questions, or locating specific content. For complex
        questions spanning multiple topics, use search_deep instead.

        RETURNS: By default returns compact results (no full text) with score,
        token_count, title, summary, and location metadata. This lets you scan
        results and decide which to fetch in full via get_context. Set
        compact=false to include full chunk text inline (costs more context).

        TIPS: Start with compact results (default). Check scores and summaries
        to identify relevant hits, then call get_context with chunk_id to read
        the full text of promising results. Use list_collections to discover
        available topics/subtopics for filtering.
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
        """Read more content around a search result or a specific section.

        WHEN TO USE: After a search returns a promising result and you need
        more context. Pass the chunk_id from the search result, or specify
        a collection + episode + time range directly.

        PARAMETERS:
        - chunk_id: Pass this from a search result's chunk_id field. The tool
          will look up the chunk's location and expand around it.
        - OR provide collection + episode_num + start_sec to target a specific
          time range directly.
        - direction: 'around' expands equally in both directions. 'before'
          fetches content leading up to the point. 'after' fetches what follows.
        - amount_sec: Total seconds of context to fetch (default 300 = 5 minutes).
        - page_tokens: Set to control how many tokens per page (e.g. 500, 1000).
          Use page param to navigate. Response includes total_pages.
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
            if chunk_id and chunk_id not in fetched_ids:
                fetched_ids.append(chunk_id)
            session = _get_session(_default_session_id)
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

    # ── get_toc ──────────────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def get_toc(
        collection: Annotated[str, Field(description="Collection ID. Use list_collections to find IDs.")],
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

    # ── list_collections ─────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def list_collections() -> dict:
        """List all indexed collections in the knowledge base.

        WHEN TO USE: Before searching, to discover what content is available
        and what topic/subtopic filters you can use. Also useful to check
        if specific content has already been ingested.

        RETURNS: Dict with total_chunks count and a list of collections.
        Each collection shows its display name, topic, subtopic, episode
        count, and episode list.
        """
        try:
            store = get_store()
            collections = store.list_collections()
            total = store.chunk_count()
            return {
                "success": True,
                "total_chunks": total,
                "collections": [
                    {
                        "collection": c["collection"],
                        "collection_display": c["collection_display"],
                        "topic": c["topic"],
                        "subtopic": c["subtopic"],
                        "episode_count": c["episode_count"],
                        "episodes": c["episodes"],
                    }
                    for c in collections
                ],
            }
        except Exception as e:
            return {"success": False, "error": str(e), "total_chunks": 0, "collections": []}

    # ── delete_collection ────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
    def delete_collection(
        collection: Annotated[str, Field(description="Collection ID to delete. Use list_collections to find IDs.")],
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

    # ── health ───────────────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def health() -> dict:
        """Check Lore server status and knowledge base stats.

        WHEN TO USE: To verify the server is running and check how much
        content is indexed before performing operations. Also useful to
        see which models are loaded and what LLM provider is active.
        """
        try:
            cfg = get_config()
            store = get_store()
            registry = get_registry()
            active = registry.active

            return {
                "success": True,
                "status": "ok",
                "version": "0.1.0",
                "embedding_model": cfg.get("embedding.model", ""),
                "reranker_model": cfg.get("search.reranker_model", ""),
                "total_chunks": store.chunk_count(),
                "active_provider": active.name if active else None,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "status": "error"}
