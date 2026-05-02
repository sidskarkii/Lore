"""Lore MCP server — exposes knowledge base tools to AI agents.

Tools (23):
    intro              — deep orientation: collections, summaries, health, workflows
    search             — hybrid search (vector + BM25 + reranking)
    search_deep        — multi-hop decomposition for complex queries
    get_context        — expand around a search result or read a section
    get_toc            — browse a collection's structure
    find_related       — cross-source connections (entity + keyword/tag + Jaccard fusion)
    entity_index       — view/rebuild the fuzzy entity index
    entity_graph       — entity co-occurrence graph (NPMI, communities, bridges)
    keyword_graph      — keyword/tag co-occurrence graph (NPMI, communities, bridges)
    reset_session      — clear fetch history after context compaction
    ingest             — auto-detect and ingest content
    ingest_status      — check ingestion progress
    rate_result        — explicit feedback on search results
    delete_collection  — remove a collection
    wiki_search        — search wiki pages (vector + FTS)
    wiki_get_page      — read a wiki page by ID or slug
    wiki_generate_page — generate or refresh a wiki page (incl. comparison)
    wiki_related       — browse wiki page connections
    wiki_claims        — inspect claim-level provenance
    wiki_queue         — manage stale/missing pages
    wiki_lint          — audit wiki health (broken provenance, orphans, gaps)
    wiki_generate_all  — recursive generation (plan/repair/expand modes)
    wiki_hierarchy     — browse page hierarchy (type indexes + topic clusters)
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
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


def _log_tool(
    session_id: str,
    tool_name: str,
    request: dict | None = None,
    result: dict | None = None,
    entities: dict | None = None,
    latency_ms: int | None = None,
):
    try:
        db = get_database()
        status = "error" if (result and result.get("success", True) is False) else "success"
        error_text = result.get("error") if result and not result.get("success", True) else None
        summary = None
        if result:
            if "total" in result:
                summary = f"{result['total']} results"
            elif "page_id" in result:
                summary = result["page_id"]
            elif "message" in result:
                summary = result["message"]
        db.log_event(
            session_id=session_id,
            tool_name=tool_name,
            request=request,
            response_summary=summary,
            entities=entities,
            status=status,
            latency_ms=latency_ms,
            error_text=error_text,
        )
    except Exception:
        pass


def _compound_wiki_pages(coll_id: str, wm) -> int:
    """Regenerate existing entity/concept pages that overlap with a newly ingested collection."""
    from ..core.wiki_generate import generate_entity_page, generate_concept_page
    from ..core.wiki import _slug
    from ..core.wiki_lint import _parse_tags
    from ..core.store import get_store

    store = get_store()
    chunks = store.get_all_chunks(coll_id)
    if not chunks:
        return 0

    try:
        from ..core.entities import get_entity_index
        ei = get_entity_index()
    except Exception:
        ei = None

    new_entity_canonicals: set[str] = set()
    new_tag_slugs: set[str] = set()
    for ch in chunks:
        ents_raw = ch.get("entities", "")
        if ents_raw:
            try:
                ents = json.loads(ents_raw) if isinstance(ents_raw, str) else ents_raw
                for e in ents:
                    if isinstance(e, dict):
                        name = e.get("name", "")
                        if ei:
                            cluster = ei.resolve(name)
                            if cluster:
                                new_entity_canonicals.add(cluster.canonical.lower())
                                continue
                        new_entity_canonicals.add(name.lower())
            except (ValueError, TypeError):
                pass
        for t in _parse_tags(ch.get("concept_tags", "")):
            s = _slug(t)
            if s:
                new_tag_slugs.add(s)

    if not new_entity_canonicals and not new_tag_slugs:
        return 0

    stale_ids = {p["page_id"] for p in wm.get_stale_pages()}
    compounded = 0

    for meta in wm.list_pages():
        pid = meta["page_id"]
        if pid in stale_ids:
            continue
        ptype = meta.get("page_type", "")
        slug = pid.split("/", 1)[-1] if "/" in pid else pid
        title = meta.get("title", slug)

        if ptype == "entity":
            canonical = title.lower()
            if ei:
                cluster = ei.resolve(title)
                if cluster:
                    canonical = cluster.canonical.lower()
            if canonical in new_entity_canonicals:
                try:
                    generate_entity_page(title, force=True)
                    compounded += 1
                except Exception:
                    pass

        elif ptype == "concept":
            if slug in new_tag_slugs:
                try:
                    generate_concept_page(slug, force=True)
                    compounded += 1
                except Exception:
                    pass

    return compounded


def _post_ingest_wiki(collection_display: str):
    """Auto-generate wiki pages after successful ingest."""
    try:
        from ..core.wiki import get_wiki_manager
        from ..core.ingest import _sanitize
        wm = get_wiki_manager()
        coll_id = _sanitize(collection_display)
        source_count = wm.generate_source_pages(collection=coll_id)
        if source_count:
            print(f"  [wiki] Auto-generated {source_count} source pages")

        try:
            from ..core.wiki_generate import generate_wiki_pages, generate_entity_page, generate_concept_page

            stale = [
                sp for sp in wm.get_stale_pages()
                if coll_id in sp.get("source_collections", [])
            ]
            refreshed = 0
            for sp in stale:
                pid = sp["page_id"]
                ptype = sp.get("page_type", "")
                slug = pid.split("/", 1)[-1] if "/" in pid else pid
                try:
                    if ptype == "entity":
                        generate_entity_page(sp.get("title", slug), force=True)
                        refreshed += 1
                    elif ptype == "concept":
                        generate_concept_page(slug, force=True)
                        refreshed += 1
                except Exception:
                    pass
            if refreshed:
                print(f"  [wiki] Refreshed {refreshed} stale pages")

            compounded = _compound_wiki_pages(coll_id, wm)
            if compounded:
                print(f"  [wiki] Compounded {compounded} overlapping pages with new source")

            concept_pages = generate_wiki_pages(page_type="concept", limit=10, min_chunks=3)
            entity_pages = generate_wiki_pages(page_type="entity", limit=10, min_chunks=2)
            total = len(concept_pages) + len(entity_pages)
            if total:
                print(f"  [wiki] Auto-generated {len(concept_pages)} concept + {len(entity_pages)} entity pages")
        except Exception as e:
            print(f"  [wiki] Skipped concept/entity generation (no LLM provider): {e}")

        try:
            from ..core.wiki_index import rebuild_fts
            rebuild_fts()
        except Exception:
            pass
    except Exception as e:
        print(f"  [wiki] Post-ingest wiki generation failed: {e}")


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
        "- entity_graph — co-occurrence graph: NPMI neighbors, topic communities, bridge entities\n"
        "- keyword_graph — keyword/tag co-occurrence: NPMI neighbors, topic clusters, bridge terms\n"
        "- reset_session — call after context compaction so fetched chunks regain full relevance\n"
        "\n"
        "Wiki tools (synthesized knowledge pages):\n"
        "- wiki_search(query) — search wiki pages for concepts, entities, or topics\n"
        "- wiki_get_page(page_id) — read a full wiki page with claims and provenance\n"
        "- wiki_generate_page(page_type, target, collections?) — generate/refresh a page; for comparison pages pass collections=[...]\n"
        "- wiki_related(page_id) — browse connections between wiki pages\n"
        "- wiki_claims(page_id) — inspect claim-level provenance without full page\n"
        "- wiki_queue() — list stale or missing pages\n"
        "\n"
        "Avoid: long multi-sentence queries, "
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

            model_status = []
            try:
                from ..core.lifecycle import get_model_manager
                mgr = get_model_manager()
                model_status = mgr.status()
                health_info["loaded_models_ram_mb"] = mgr.loaded_ram_mb()
            except Exception:
                pass

            return {
                "success": True,
                "overview": {
                    "total_chunks": total_chunks,
                    "total_collections": len(collections),
                    "topics": {t: len(names) for t, names in topic_map.items()},
                    "top_tags": [t for t, _ in top_tags],
                },
                "health": health_info,
                "models": model_status,
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
        expand_query: Annotated[bool, Field(default=False, description="LLM-rewrite the query into variant phrasings for better recall. Requires an LLM provider (auto-detected when running under Claude Code).")] = False,
    ) -> dict:
        """Step 1: Search the knowledge base. Returns compact results (~50 tokens each).

        WHEN TO USE: Primary tool for finding information. Use for factual
        queries, how-to questions, or locating specific content. Keep queries
        short and specific. For complex multi-topic questions, use search_deep.

        RETURNS: Compact metadata only — score, title, summary, keywords,
        location. Scan these, then call get_context for full text of
        promising hits.
        """
        try:
            engine = get_search_engine()
            results = engine.search(
                query=query,
                n_results=n_results,
                topic=topic,
                subtopic=subtopic,
                expand=False,
                session_id=_default_session_id,
                _force_expansion=expand_query,
            )
            formatter = _format_compact_result
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

            response = {
                "success": True,
                "query": query,
                "total": len(results),
                "results": formatted,
            }

            try:
                from ..core.query_intent import detect_query_intent
                intent = detect_query_intent(query)
                if intent.wiki_favorable and not intent.chunk_favorable:
                    response["wiki_hint"] = intent.suggested_hint
            except Exception:
                pass

            return response
        except Exception as e:
            return {"success": False, "error": str(e), "query": query, "total": 0, "results": []}

    # ── search_deep ──────────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def search_deep(
        query: Annotated[str, Field(description="Complex search query that may span multiple topics.")],
        n_results: Annotated[int, Field(default=5, ge=1, le=20, description="Number of results to return.")] = 5,
        topic: Annotated[str | None, Field(default=None, description="Filter by topic.")] = None,
        subtopic: Annotated[str | None, Field(default=None, description="Filter by subtopic.")] = None,
    ) -> dict:
        """Deep search using multi-hop query decomposition.

        Breaks complex questions into simpler sub-queries using an LLM,
        runs hybrid search for each, then fuses and reranks all results.
        Slower than search (makes LLM calls) but better for questions that
        need information from multiple sources.

        WHEN TO USE: For complex questions spanning multiple topics.
        Falls back to regular search if no LLM provider is configured.

        RETURNS: Compact metadata only, same format as search.
        Use get_context to fetch full text for promising results.

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
            formatted = [_format_compact_result(r) for r in results]

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
                    _ingest_jobs[job_id]["chunks"] = chunks
                    _ingest_jobs[job_id]["message"] = f"Ingested {chunks} chunks, generating wiki pages…"

                await asyncio.to_thread(_post_ingest_wiki, name)

                with _ingest_jobs_lock:
                    _ingest_jobs[job_id]["status"] = "done"
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
        chunk_id: Annotated[str | None, Field(default=None, description="Find chunks related to this chunk via shared entities, keywords, and tags (fused Dice scoring).")] = None,
        entity: Annotated[str | None, Field(default=None, description="Find chunks mentioning this entity (fuzzy-matched). Entity-only mode.")] = None,
        collection: Annotated[str | None, Field(default=None, description="Limit results to this collection.")] = None,
        n_results: Annotated[int, Field(default=10, ge=1, le=50, description="Number of results.")] = 10,
    ) -> dict:
        """Step 4: Discover cross-source connections.

        Two modes:
        - chunk_id: Fused Dice scoring (0.60*entity + 0.25*keyword + 0.15*tag).
          Uses postings-based candidate generation — only scores chunks that
          share at least one entity, keyword, or tag. Fast even at large scale.
        - entity: Entity-only lookup via postings. Finds all chunks mentioning
          a specific entity, with fuzzy name matching.

        WHEN TO USE: After finding a useful chunk, discover what other
        sources say about the same topics.
        """
        from ..core.cross_index import get_cross_index

        try:
            cx = get_cross_index()

            if entity and not chunk_id:
                return cx.find_by_entity(entity, collection=collection, n_results=n_results)

            if chunk_id:
                return cx.find_related(chunk_id, collection=collection, n_results=n_results)

            return {"success": False, "error": "Provide chunk_id or entity name"}
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

    # ── entity_graph ────────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def entity_graph(
        entity: Annotated[str | None, Field(default=None, description="Entity to inspect (fuzzy-matched).")] = None,
        mode: Annotated[str, Field(default="neighbors", description="Query mode: 'neighbors' (top NPMI connections), 'community' (same topic cluster), 'bridges' (entities connecting communities), 'stats' (graph overview).")] = "neighbors",
        n_results: Annotated[int, Field(default=10, ge=1, le=50, description="Number of results.")] = 10,
        rebuild: Annotated[bool, Field(default=False, description="Force rebuild the co-occurrence graph.")] = False,
    ) -> dict:
        """Query the entity co-occurrence graph.

        Entities that appear together in chunks are connected by edges
        weighted with NPMI (statistically surprising associations).
        Louvain community detection groups entities into topic clusters.

        WHEN TO USE: To explore entity relationships beyond simple
        co-mention. Find what entities are statistically associated,
        which topic clusters exist, and which entities bridge different
        communities.

        Modes:
        - neighbors: top connections for an entity, ranked by NPMI
        - community: all entities in the same topic cluster
        - bridges: entities that connect different communities
        - stats: graph overview (nodes, edges, communities)
        """
        from ..core.graph import get_entity_graph

        try:
            g = get_entity_graph(rebuild=rebuild)

            if mode == "stats":
                return {"success": True, **g.stats()}

            if mode == "bridges":
                return {"success": True, "bridges": g.bridges(n_results)}

            if not entity:
                return {"success": False, "error": "Provide entity name for neighbors/community mode"}

            if mode == "neighbors":
                results = g.neighbors(entity, n_results)
                return {"success": True, "entity": entity, "neighbors": results}

            if mode == "community":
                members = g.community_members(entity)
                return {"success": True, "entity": entity, "community_members": members}

            return {"success": False, "error": f"Unknown mode: {mode}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── keyword_graph ──────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def keyword_graph(
        term: Annotated[str | None, Field(default=None, description="Keyword or concept tag to inspect. Can use namespace prefix (kw:X or tag:Y) or bare term.")] = None,
        mode: Annotated[str, Field(default="neighbors", description="Query mode: 'neighbors' (top NPMI connections), 'community' (same topic cluster), 'bridges' (terms connecting communities), 'stats' (graph overview).")] = "neighbors",
        n_results: Annotated[int, Field(default=10, ge=1, le=50, description="Number of results.")] = 10,
        rebuild: Annotated[bool, Field(default=False, description="Force rebuild the keyword/tag co-occurrence graph.")] = False,
    ) -> dict:
        """Query the keyword/tag co-occurrence graph.

        Keywords and concept tags that appear together in chunks are
        connected by NPMI-weighted edges. Nodes are namespaced (kw:X for
        keywords, tag:Y for concept tags). Louvain community detection
        groups related terms into topic clusters.

        WHEN TO USE: To explore topical structure of the knowledge base.
        Find what keywords and tags are statistically associated, discover
        topic clusters, and find bridge terms connecting different domains.

        Modes:
        - neighbors: top NPMI connections for a term
        - community: all terms in the same topic cluster
        - bridges: terms that connect different communities
        - stats: graph overview (nodes, edges, keyword/tag counts)
        """
        from ..core.graph import get_keyword_graph

        try:
            g = get_keyword_graph(rebuild=rebuild)

            if mode == "stats":
                return {"success": True, **g.stats()}

            if mode == "bridges":
                return {"success": True, "bridges": g.bridges(n_results)}

            if not term:
                return {"success": False, "error": "Provide term for neighbors/community mode"}

            if mode == "neighbors":
                results = g.neighbors(term, n_results)
                return {"success": True, "term": term, "neighbors": results}

            if mode == "community":
                members = g.community_members(term)
                return {"success": True, "term": term, "community_members": members}

            return {"success": False, "error": f"Unknown mode: {mode}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── wiki_search ────────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def wiki_search(
        query: Annotated[str, Field(description="Search query for wiki pages.")],
        page_type: Annotated[str | None, Field(default=None, description="Filter by page type: entity, concept, source, comparison.")] = None,
        n_results: Annotated[int, Field(default=8, ge=1, le=30, description="Number of results.")] = 8,
        include_stale: Annotated[bool, Field(default=False, description="Include stale pages in results.")] = False,
    ) -> dict:
        """Search wiki pages for synthesized knowledge.

        WHEN TO USE: When you want a synthesized overview of a concept,
        entity, or topic rather than raw source chunks. Wiki pages
        aggregate and verify claims across multiple sources.

        Returns compact results with page_id, title, page_type,
        confidence, corroboration, and a text preview.
        """
        t0 = time.perf_counter()
        try:
            from ..core.wiki_index import search_wiki
            results = search_wiki(
                query=query,
                page_type=page_type,
                n_results=n_results,
                include_stale=include_stale,
            )
            result = {
                "success": True,
                "total": len(results),
                "results": results,
            }
            _log_tool(_default_session_id, "wiki_search",
                      request={"query": query, "page_type": page_type},
                      result=result,
                      entities={"page_ids": [r.get("page_id", "") for r in results]},
                      latency_ms=int((time.perf_counter() - t0) * 1000))
            return result
        except Exception as e:
            err = {"success": False, "error": str(e)}
            _log_tool(_default_session_id, "wiki_search",
                      request={"query": query, "page_type": page_type},
                      result=err,
                      latency_ms=int((time.perf_counter() - t0) * 1000))
            return err

    # ── wiki_get_page ──────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def wiki_get_page(
        page_id: Annotated[str | None, Field(default=None, description="Page ID like 'concept/deception' or 'entity/sun-tzu'.")] = None,
        slug: Annotated[str | None, Field(default=None, description="Page slug to search for across page types.")] = None,
        include_content: Annotated[bool, Field(default=True, description="Include full prose content. Set False for claims-only mode (saves tokens).")] = True,
        include_claims: Annotated[bool, Field(default=True, description="Include structured claim data.")] = True,
        include_provenance: Annotated[bool, Field(default=True, description="Include full provenance chunk IDs.")] = True,
    ) -> dict:
        """Read a wiki page — synthesized knowledge with claim-level provenance.

        WHEN TO USE: After wiki_search finds a relevant page, use this to
        read the full content with claims, verification status, and source
        chunk IDs. Each claim is tagged supported/partially_supported/review/conflicted.

        Set include_content=False for claims-only mode — returns structured
        claims without prose, cutting response tokens roughly in half.

        Provide either page_id (exact) or slug (searches all page types).
        """
        t0 = time.perf_counter()
        req = {"page_id": page_id, "slug": slug, "include_content": include_content}
        try:
            from ..core.wiki import get_wiki_manager
            wm = get_wiki_manager()

            if not page_id and slug:
                for ptype in ("concept", "entity", "source", "comparison"):
                    candidate = f"{ptype}/{slug}"
                    if wm.page_exists(candidate):
                        page_id = candidate
                        break

            if not page_id:
                err = {"success": False, "error": "Provide page_id or slug"}
                _log_tool(_default_session_id, "wiki_get_page", request=req, result=err,
                          latency_ms=int((time.perf_counter() - t0) * 1000))
                return err

            page = wm.get_page(page_id)
            if not page:
                err = {"success": False, "error": f"Page not found: {page_id}"}
                _log_tool(_default_session_id, "wiki_get_page", request=req, result=err,
                          latency_ms=int((time.perf_counter() - t0) * 1000))
                return err

            result = {
                "success": True,
                "page_id": page.page_id,
                "page_type": page.page_type,
                "title": page.title,
                "status": page.status,
                "version": page.version,
                "source_collections": page.source_collections,
                "source_chunk_count": page.source_chunk_count,
                "supporting_source_count": page.supporting_source_count,
                "corroboration_level": page.corroboration_level,
                "confidence": page.confidence,
                "related_pages": page.related_pages,
                "backlinks": wm.get_backlinks(page_id),
            }

            if include_content:
                result["content"] = page.content

            if include_claims and page.generation:
                result["claims"] = page.generation.get("claims", [])
                result["claim_count"] = page.generation.get("claim_count", 0)

            if include_provenance and page.generation:
                result["generation_strategy"] = page.generation.get("strategy", "")
                result["inputs_hash"] = page.generation.get("inputs_hash", "")
                claims = page.generation.get("claims", [])
                chunk_ids = []
                for c in claims:
                    chunk_ids.extend(c.get("chunk_ids", []))
                result["provenance_chunk_ids"] = sorted(set(chunk_ids))

            _log_tool(_default_session_id, "wiki_get_page", request=req, result=result,
                      entities={"page_id": page_id, "chunk_ids": result.get("provenance_chunk_ids", [])},
                      latency_ms=int((time.perf_counter() - t0) * 1000))
            return result
        except Exception as e:
            err = {"success": False, "error": str(e)}
            _log_tool(_default_session_id, "wiki_get_page", request=req, result=err,
                      latency_ms=int((time.perf_counter() - t0) * 1000))
            return err

    # ── wiki_generate_page ─────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def wiki_generate_page(
        page_type: Annotated[str, Field(description="Page type: entity, concept, source, or comparison.")],
        target: Annotated[str, Field(description="Entity name, concept tag, collection name, or comparison topic.")],
        collections: Annotated[list[str] | None, Field(default=None, description="For comparison pages: list of 2-4 collection names to compare.")] = None,
        force: Annotated[bool, Field(default=False, description="Force regeneration even if page exists and is not stale.")] = False,
    ) -> dict:
        """Generate or refresh a wiki page on demand.

        WHEN TO USE: When a concept or entity deserves a synthesized
        page but one doesn't exist yet, or when a page is stale.
        For comparisons, use page_type='comparison' with target as the
        topic and collections as the sources to compare.

        PARAMETERS:
        - page_type: 'entity', 'concept', 'source', or 'comparison'
        - target: the name/tag/topic to generate for
        - collections: required for comparison — 2-4 collection names
        - force: regenerate even if current page is fresh
        """
        t0 = time.perf_counter()
        req = {"page_type": page_type, "target": target, "collections": collections}
        try:
            from ..core.wiki_generate import generate_entity_page, generate_concept_page, generate_comparison_page
            from ..core.wiki import get_wiki_manager
            if page_type == "entity":
                page = generate_entity_page(target, force=force)
            elif page_type == "concept":
                page = generate_concept_page(target, force=force)
            elif page_type == "source":
                wm = get_wiki_manager()
                count = wm.generate_source_pages(collection=target, force=force)
                result = {"success": True, "message": f"Generated/refreshed {count} source pages"}
                _log_tool(_default_session_id, "wiki_generate_page", request=req, result=result,
                          latency_ms=int((time.perf_counter() - t0) * 1000))
                return result
            elif page_type == "comparison":
                if not collections or len(collections) < 2:
                    err = {"success": False, "error": "Comparison requires 'collections' with 2-4 collection names."}
                    _log_tool(_default_session_id, "wiki_generate_page", request=req, result=err,
                              latency_ms=int((time.perf_counter() - t0) * 1000))
                    return err
                page = generate_comparison_page(target, collections, force=force)
            else:
                err = {"success": False, "error": f"Unknown page_type: {page_type}. Use entity, concept, source, or comparison."}
                _log_tool(_default_session_id, "wiki_generate_page", request=req, result=err,
                          latency_ms=int((time.perf_counter() - t0) * 1000))
                return err

            if not page:
                err = {"success": False, "error": f"Not enough evidence to generate {page_type} page for '{target}'"}
                _log_tool(_default_session_id, "wiki_generate_page", request=req, result=err,
                          latency_ms=int((time.perf_counter() - t0) * 1000))
                return err

            result = {
                "success": True,
                "page_id": page.page_id,
                "title": page.title,
                "claim_count": page.generation.get("claim_count", 0),
                "corroboration_level": page.corroboration_level,
                "confidence": page.confidence,
                "source_collections": page.source_collections,
            }
            dropped = page.generation.get("dropped_collections", [])
            if dropped:
                result["dropped_collections"] = dropped
                result["note"] = f"Collections excluded (insufficient evidence on '{target}'): {', '.join(dropped)}"
            _log_tool(_default_session_id, "wiki_generate_page", request=req, result=result,
                      entities={"page_id": result.get("page_id", "")},
                      latency_ms=int((time.perf_counter() - t0) * 1000))
            return result
        except Exception as e:
            err = {"success": False, "error": str(e)}
            _log_tool(_default_session_id, "wiki_generate_page", request=req, result=err,
                      latency_ms=int((time.perf_counter() - t0) * 1000))
            return err

    # ── wiki_related ───────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def wiki_related(
        page_id: Annotated[str, Field(description="Page ID to find relations for.")],
        relation: Annotated[str, Field(default="all", description="Relation type: all, concepts, entities, sources, backlinks.")] = "all",
        n_results: Annotated[int, Field(default=10, ge=1, le=50, description="Max results.")] = 10,
    ) -> dict:
        """Browse the wiki graph — related pages, backlinks, connections.

        WHEN TO USE: After reading a wiki page, explore what else
        links to it or what it links to.
        """
        from ..core.wiki import get_wiki_manager

        try:
            wm = get_wiki_manager()
            page = wm.get_page(page_id)
            if not page:
                return {"success": False, "error": f"Page not found: {page_id}"}

            result = {"success": True, "page_id": page_id}

            if relation in ("all", "backlinks"):
                result["backlinks"] = wm.get_backlinks(page_id)

            related = page.related_pages or []
            if relation == "all":
                result["related_pages"] = related[:n_results]
            elif relation == "concepts":
                result["related_pages"] = [r for r in related if r.startswith("concept/")][:n_results]
            elif relation == "entities":
                result["related_pages"] = [r for r in related if r.startswith("entity/")][:n_results]
            elif relation == "sources":
                result["related_pages"] = [r for r in related if r.startswith("source/")][:n_results]

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── wiki_claims ────────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def wiki_claims(
        page_id: Annotated[str, Field(description="Page ID to inspect claims for.")],
        min_support: Annotated[int, Field(default=1, ge=0, description="Minimum support count to include.")] = 1,
        status_filter: Annotated[str | None, Field(default=None, description="Filter by status: supported, partially_supported, review, conflicted.")] = None,
        n_results: Annotated[int, Field(default=20, ge=1, le=100, description="Max claims to return.")] = 20,
    ) -> dict:
        """Inspect claim-level provenance without reading the full page.

        WHEN TO USE: When you need to verify what evidence supports
        specific claims, or find which claims need review.
        Each claim includes chunk_ids, collections, support count,
        verification status, and corroboration level.
        """
        from ..core.wiki import get_wiki_manager

        try:
            wm = get_wiki_manager()
            page = wm.get_page(page_id)
            if not page:
                return {"success": False, "error": f"Page not found: {page_id}"}

            claims = page.generation.get("claims", []) if page.generation else []

            if status_filter:
                claims = [c for c in claims if c.get("status") == status_filter]
            if min_support > 0:
                claims = [c for c in claims if c.get("support_count", 0) >= min_support]

            claims = claims[:n_results]

            by_status = {}
            all_claims = page.generation.get("claims", []) if page.generation else []
            for c in all_claims:
                s = c.get("status", "unknown")
                by_status[s] = by_status.get(s, 0) + 1

            return {
                "success": True,
                "page_id": page_id,
                "total_claims": len(all_claims),
                "status_summary": by_status,
                "returned": len(claims),
                "claims": claims,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── wiki_queue ─────────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def wiki_queue(
        action: Annotated[str, Field(default="list", description="Action: list (show stale/missing), rebuild_index (reindex all pages), stats.")] = "list",
        limit: Annotated[int, Field(default=20, ge=1, le=100, description="Max items to return.")] = 20,
    ) -> dict:
        """Manage wiki page queue — stale pages, rebuild index, stats.

        WHEN TO USE: To check wiki health, find pages needing refresh,
        or rebuild the search index after manual edits.

        Actions:
        - list: show stale pages that need regeneration
        - rebuild_index: rebuild the LanceDB wiki search index from all pages
        - stats: page counts by type and status
        """
        from ..core.wiki import get_wiki_manager

        try:
            wm = get_wiki_manager()

            if action == "stats":
                return {"success": True, **wm.stats()}

            if action == "rebuild_index":
                from ..core.wiki_index import rebuild_index
                count = rebuild_index()
                return {"success": True, "message": f"Rebuilt wiki index: {count} fragments indexed"}

            if action == "list":
                stale = wm.get_stale_pages()
                return {
                    "success": True,
                    "stale_count": len(stale),
                    "stale_pages": stale[:limit],
                }

            return {"success": False, "error": f"Unknown action: {action}. Use list, rebuild_index, or stats."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── wiki_lint ──────────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def wiki_lint(
        checks: Annotated[list[str] | None, Field(default=None, description="Checks to run. Default: all. Options: stale, orphan, weak_claims, claim_summary, broken_links, broken_provenance, generation_drift, source_gaps.")] = None,
    ) -> dict:
        """Audit wiki health — find broken provenance, orphan pages, weak claims, source gaps.

        WHEN TO USE: After ingesting new content, after wiki generation,
        or periodically to check wiki integrity. Returns findings grouped
        by severity (error/warning/info).

        Checks:
        - stale: pages needing regeneration
        - orphan: pages with no incoming links
        - weak_claims: claims with review/conflicted status or low support
        - claim_summary: distribution of claim verification statuses
        - broken_links: related_pages pointing to nonexistent pages
        - broken_provenance: claims referencing deleted chunks
        - generation_drift: claim_count mismatches, empty content, missing metadata
        - source_gaps: entities/concepts spanning multiple sources without wiki pages
        """
        try:
            from ..core.wiki_lint import lint_wiki
            return {"success": True, **lint_wiki(checks=checks)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── wiki_generate_all ─────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def wiki_generate_all(
        mode: Annotated[str, Field(description="Mode: 'plan' (dry-run with cost estimate), 'repair' (broken links only), 'expand' (full candidate pool).")],
        limit: Annotated[int, Field(default=25, ge=1, le=100, description="Max pages to generate per invocation.")] = 25,
        min_chunks_concept: Annotated[int, Field(default=3, ge=1, description="Min evidence chunks for concept pages.")] = 3,
        min_chunks_entity: Annotated[int, Field(default=2, ge=1, description="Min evidence chunks for entity pages.")] = 2,
    ) -> dict:
        """Recursive wiki generation — discover and write missing pages.

        WHEN TO USE: After ingesting content, to fill wiki gaps. Use 'plan'
        first to see what would be generated and the estimated cost, then
        'repair' to fix broken links, or 'expand' for full coverage.

        Modes:
        - plan: dry-run returning ranked candidates, scores, and LLM cost estimate.
          Shows what 'expand' would do. Use plan with limit to preview repair scope too.
        - repair: generate only pages needed to resolve broken links
        - expand: generate from the full candidate pool (broken links + new discoveries)

        Candidates are ranked by: link pressure (40%), evidence count (25%),
        source diversity (20%), graph centrality (15%). Pages are generated
        in batch waves up to the limit.
        """
        try:
            from ..core.wiki_candidates import plan as wiki_plan, discover_candidates
            from ..core.wiki_generate import generate_entity_page, generate_concept_page
            from ..core.wiki import get_wiki_manager

            if mode in ("plan", "plan_repair"):
                repair_only = (mode == "plan_repair")
                result = wiki_plan(
                    repair_only=repair_only, limit=limit,
                    min_chunks_concept=min_chunks_concept,
                    min_chunks_entity=min_chunks_entity,
                )
                return {"success": True, "mode": mode, **result}

            if mode not in ("repair", "expand"):
                return {"success": False, "error": f"Unknown mode: {mode}. Use plan, plan_repair, repair, or expand."}

            wm = get_wiki_manager()
            existing_before = {m["page_id"] for m in wm.list_pages()}

            repair_only = (mode == "repair")
            candidates = discover_candidates(
                repair_only=repair_only,
                min_chunks_concept=min_chunks_concept,
                min_chunks_entity=min_chunks_entity,
            )[:limit]

            created = []
            skipped = []
            failed = []
            for c in candidates:
                try:
                    if c.page_type == "entity":
                        page = generate_entity_page(c.target, force=False)
                    elif c.page_type == "concept":
                        page = generate_concept_page(c.target, force=False)
                    else:
                        continue
                    if not page:
                        failed.append({"target": c.target, "reason": "insufficient evidence"})
                    elif page.page_id in existing_before:
                        skipped.append({"page_id": page.page_id, "reason": "already exists"})
                    else:
                        created.append({
                            "page_id": page.page_id,
                            "title": page.title,
                            "claim_count": page.generation.get("claim_count", 0),
                            "score": round(c.score, 4),
                        })
                except Exception as e:
                    failed.append({"target": c.target, "reason": str(e)[:100]})

            try:
                from ..core.wiki_index import rebuild_fts
                rebuild_fts()
            except Exception:
                pass

            return {
                "success": True,
                "mode": mode,
                "created": len(created),
                "skipped": len(skipped),
                "failed": len(failed),
                "pages": created,
                "skipped_details": skipped[:10],
                "failures": failed[:10],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── wiki_hierarchy ────────────────────────────────────────────

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def wiki_hierarchy(
        action: Annotated[str, Field(default="browse", description="Action: 'browse' (show hierarchy), 'rebuild' (recompute clusters), 'cluster' (show one cluster by ID).")] = "browse",
        cluster_id: Annotated[int | None, Field(default=None, description="Cluster ID to inspect (for action='cluster').")] = None,
    ) -> dict:
        """Browse the wiki page hierarchy — type indexes and topic clusters.

        WHEN TO USE: To understand how wiki pages are organized, find
        related pages within topic groups, or navigate the wiki structure.

        Actions:
        - browse: show type indexes and topic cluster summaries
        - rebuild: recompute clusters from current page data
        - cluster: show full details for a specific topic cluster
        """
        try:
            from ..core.wiki_hierarchy import build_hierarchy, load_hierarchy

            if action == "rebuild":
                h = build_hierarchy()
                return {"success": True, "action": "rebuild", **h}

            if action == "cluster":
                h = load_hierarchy()
                if not h:
                    h = build_hierarchy()
                if cluster_id is None:
                    return {"success": False, "error": "Provide cluster_id for action='cluster'."}
                for c in h.get("topic_clusters", []):
                    if c["cluster_id"] == cluster_id:
                        return {"success": True, "action": "cluster", **c}
                return {"success": False, "error": f"Cluster {cluster_id} not found."}

            if action == "browse":
                h = load_hierarchy()
                if not h:
                    h = build_hierarchy()

                summary = {
                    "total_pages": h["total_pages"],
                    "total_clusters": h["total_clusters"],
                    "type_indexes": {t: len(pages) for t, pages in h["type_indexes"].items()},
                    "clusters": [
                        {"cluster_id": c["cluster_id"], "label": c["label"], "page_count": c["page_count"]}
                        for c in h["topic_clusters"]
                    ],
                }
                return {"success": True, "action": "browse", **summary}

            return {"success": False, "error": f"Unknown action: {action}. Use browse, rebuild, or cluster."}
        except Exception as e:
            return {"success": False, "error": str(e)}

