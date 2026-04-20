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

import re
from typing import Annotated

from mcp.server.fastmcp import FastMCP, Context
from mcp.types import ToolAnnotations
from pydantic import Field

from ..core.config import get_config
from ..core.search import get_search_engine
from ..core.store import get_store
from ..providers.registry import get_registry


def create_mcp_server() -> FastMCP:
    mcp = FastMCP(
        "Lore",
        instructions=(
            "Lore is a local-first RAG knowledge base. It indexes videos, documents, "
            "code, and web pages into searchable chunks with timestamp-level precision. "
            "Use 'search' for most queries — results are compact by default (metadata "
            "only, no full text). Scan the scores, summaries, and token_counts, then "
            "call 'get_context' with chunk_id to fetch full text for the results you "
            "actually need. Use 'search_deep' for complex questions spanning multiple "
            "topics. Use 'list_collections' to see what's indexed before searching."
        ),
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
        "entities": r.get("entities", ""),
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
            )
            formatter = _format_compact_result if compact else _format_result
            return {
                "success": True,
                "query": query,
                "total": len(results),
                "compact": compact,
                "results": [formatter(r) for r in results],
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
            return {
                "success": True,
                "query": query,
                "total": len(results),
                "compact": compact,
                "results": [formatter(r) for r in results],
            }
        except Exception as e:
            return {"success": False, "error": str(e), "query": query, "total": 0, "results": []}

    # ── get_context ──────────────────────────────────────────────────

    _CHUNK_ID_RE = re.compile(r"^(.+)_ep(\d{3})_\d{4}$")

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def get_context(
        chunk_id: Annotated[str | None, Field(default=None, description="Chunk ID from a search result. Provide this OR collection+episode_num+start_sec.")] = None,
        collection: Annotated[str | None, Field(default=None, description="Collection name.")] = None,
        episode_num: Annotated[int | None, Field(default=None, description="Episode number within the collection.")] = None,
        start_sec: Annotated[int | None, Field(default=None, description="Start of the time window in seconds.")] = None,
        end_sec: Annotated[int | None, Field(default=None, description="End of the time window in seconds.")] = None,
        direction: Annotated[str, Field(default="around", description="Direction to expand: 'before', 'after', or 'around' the target.")] = "around",
        amount_sec: Annotated[int, Field(default=300, ge=30, le=1800, description="How many seconds of context to fetch.")] = 300,
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
        """
        try:
            store = get_store()

            if chunk_id:
                chunk = store.get_chunk_by_id(chunk_id)
                if not chunk:
                    return {"success": False, "error": f"Chunk not found: {chunk_id}"}

                m = _CHUNK_ID_RE.match(chunk_id)
                if not m:
                    return {"success": False, "error": f"Invalid chunk_id format: {chunk_id}"}

                collection = m.group(1)
                episode_num = int(m.group(2))
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

            chunks = [_format_result(row) for row in neighbors]

            result = {
                "success": True,
                "collection": collection,
                "episode_num": episode_num,
                "direction": direction,
                "total": len(chunks),
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

    # ── ingest ───────────────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
    async def ingest(
        source: Annotated[str, Field(description="YouTube URL, web page URL, file path, or folder path to ingest.")],
        name: Annotated[str, Field(description="Display name for the collection (e.g. 'Blender Donut Tutorial').")],
        topic: Annotated[str, Field(default="", description="Topic category (e.g. '3d', 'ai', 'code').")] = "",
        subtopic: Annotated[str, Field(default="", description="Subtopic (e.g. 'blender', 'python').")] = "",
        ctx: Context | None = None,
    ) -> dict:
        """Ingest content into the knowledge base.

        Auto-detects the source type:
        - YouTube URL (youtube.com or youtu.be) -> downloads transcript/audio
        - Web URL (http/https) -> extracts article content
        - Directory path -> ingests all video/audio files in folder
        - File path -> ingests document (PDF, EPUB, markdown, code, audio/video)

        WHEN TO USE: When the user wants to add new content to their knowledge
        base. This is a potentially long-running operation (especially for
        video playlists). Progress updates are reported during ingestion.

        RETURNS: Dict with success status and number of chunks indexed.
        """
        import asyncio
        from pathlib import Path
        from ..core.ingest import Ingester, IngestionProgress

        try:
            ingester = Ingester()
            loop = asyncio.get_event_loop()

            def on_progress(p: IngestionProgress):
                if ctx:
                    try:
                        future = asyncio.run_coroutine_threadsafe(
                            ctx.report_progress(
                                progress=p.completed_items,
                                total=p.total_items,
                            ),
                            loop,
                        )
                        future.result(timeout=5)
                    except Exception:
                        pass

            is_youtube = bool(re.search(r"(youtube\.com|youtu\.be)", source))
            is_url = source.startswith("http://") or source.startswith("https://")
            is_dir = Path(source).is_dir()
            is_file = Path(source).is_file()

            if is_youtube:
                chunks = await asyncio.to_thread(
                    ingester.ingest_youtube,
                    url=source, name=name, topic=topic, subtopic=subtopic,
                    on_progress=on_progress,
                )
            elif is_url:
                chunks = await asyncio.to_thread(
                    ingester.ingest_url,
                    url=source, name=name, topic=topic, subtopic=subtopic,
                    on_progress=on_progress,
                )
            elif is_dir:
                chunks = await asyncio.to_thread(
                    ingester.ingest_folder,
                    folder=source, name=name, topic=topic, subtopic=subtopic,
                    on_progress=on_progress,
                )
            elif is_file:
                chunks = await asyncio.to_thread(
                    ingester.ingest_file,
                    path=source, name=name, topic=topic, subtopic=subtopic,
                    on_progress=on_progress,
                )
            else:
                return {"success": False, "error": f"Source not found or unrecognized: {source}"}

            return {"success": True, "chunks": chunks, "message": f"Ingested {chunks} chunks from {source}"}

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
