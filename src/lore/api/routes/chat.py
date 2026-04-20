"""Chat endpoint — RAG-grounded conversation with streaming support.

POST /api/chat         — full response (non-streaming, useful for testing)
POST /api/chat/stream  — SSE streaming (used by the frontend)
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, HTTPException

from ..schemas import ChatRequest, ChatResponse, ChatMessage, SearchResult
from ...core.database import get_database
from ...core.search import get_search_engine
from ...providers.registry import get_registry


router = APIRouter(tags=["chat"])

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant with access to a curated library of tutorials "
    "and documentation. Answer the user's question based on the provided sources. "
    "Be specific — mention exact tool names, menu paths, shortcuts, and node names "
    "when they appear in the sources.\n\n"
    "At the end of your answer, list the sources with their timestamps so the user "
    "can jump directly to that moment in the video."
)


def _resolve_provider(req: ChatRequest):
    """Get the provider instance for this request."""
    registry = get_registry()
    if req.provider:
        provider = registry.get(req.provider)
        if not provider:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {req.provider}")
        return provider
    provider = registry.active
    if not provider:
        raise HTTPException(status_code=400, detail="No active provider configured")
    return provider


def _last_user_message(messages: list[ChatMessage]) -> str:
    """Extract the last user message from the conversation."""
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _search_sources(query: str, req: ChatRequest, provider=None) -> list[dict]:
    """Run hybrid search, with multi-hop fallback."""
    try:
        engine = get_search_engine()
        if req.multi_hop:
            try:
                return engine.search_multi_hop(
                    query, provider=provider,
                    n_results=req.n_sources, topic=req.topic, subtopic=req.subtopic,
                )
            except Exception:
                pass
        return engine.search(query, n_results=req.n_sources, topic=req.topic, subtopic=req.subtopic)
    except Exception:
        return []


def _sources_to_results(sources: list[dict]) -> list[SearchResult]:
    """Convert raw source dicts to SearchResult schema objects."""
    return [
        SearchResult(
            text=s.get("text", "")[:500],
            collection=s.get("collection", ""),
            collection_display=s.get("collection_display", ""),
            episode_num=s.get("episode_num", 0),
            episode_title=s.get("episode_title", ""),
            timestamp=s.get("timestamp", "00:00"),
            start_sec=s.get("start_sec", 0),
            end_sec=s.get("end_sec", 0),
            url=s.get("url", ""),
            topic=s.get("topic", ""),
            subtopic=s.get("subtopic", ""),
        )
        for s in sources
    ]


def _sources_for_db(sources: list[dict]) -> list[dict]:
    """Slim down sources for SQLite storage (no full text, just refs)."""
    return [
        {
            "collection_display": s.get("collection_display", ""),
            "episode_title": s.get("episode_title", ""),
            "timestamp": s.get("timestamp", "00:00"),
            "url": s.get("url", ""),
        }
        for s in sources
    ]


def _get_or_create_session(req: ChatRequest, provider_name: str, model: str | None) -> str:
    """Resume existing session or create a new one. Returns session_id."""
    db = get_database()
    if req.session_id:
        session = db.get_session(req.session_id)
        if session:
            return req.session_id
    title = "New Chat"
    for m in req.messages:
        if m.role == "user":
            title = m.content[:80]
            break
    session = db.create_session(title=title, provider=provider_name, model=model)
    return session["id"]


def _build_rag_messages(
    messages: list[ChatMessage],
    sources: list[dict],
) -> list[dict]:
    """Build the message list with RAG context injected."""
    context_parts = []
    for i, s in enumerate(sources, 1):
        ts = s.get("timestamp", "00:00")
        title = s.get("episode_title", "")
        collection = s.get("collection_display", "")
        url = s.get("url", "")
        text = s.get("text", "")[:800]
        context_parts.append(
            f"[Source {i}] {collection} — {title}\n"
            f"Timestamp: {ts} | {url}\n\n"
            f"{text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    result = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages[:-1]:
        result.append({"role": m.role, "content": m.content})

    last = messages[-1]
    result.append({
        "role": "user",
        "content": f"{last.content}\n\n---\nSources:\n\n{context}",
    })
    return result


@router.post(
    "/api/chat",
    response_model=ChatResponse,
    summary="Chat with RAG context",
    description=(
        "Searches the knowledge base for relevant sources, then sends "
        "the query + sources to the active LLM provider for a grounded answer."
    ),
)
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    provider = _resolve_provider(req)
    query = _last_user_message(req.messages)
    sources = _search_sources(query, req, provider)
    rag_messages = _build_rag_messages(req.messages, sources)

    try:
        answer = provider.chat(rag_messages, model=req.model)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Provider error: {e}")

    db = get_database()
    session_id = _get_or_create_session(req, provider.name, req.model)
    db.add_message(session_id, "user", query)
    db.add_message(session_id, "assistant", answer, sources=_sources_for_db(sources))

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=_sources_to_results(sources),
        provider=provider.name,
        model=req.model or "default",
    )


@router.post(
    "/api/chat/stream",
    summary="Stream chat response via SSE",
    description=(
        "Same pipeline as /api/chat but returns a Server-Sent Events stream. "
        "Events: source (retrieved sources), session (session ID), token "
        "(response chunks), done (completion)."
    ),
)
async def chat_stream_sse(req: ChatRequest):
    from sse_starlette.sse import EventSourceResponse

    provider = _resolve_provider(req)
    query = _last_user_message(req.messages)

    async def event_generator():
        t0 = time.time()
        try:
            print(f"  [chat/sse] query: {query[:80]!r}")

            if req.multi_hop:
                yield {"event": "status", "data": "Decomposing query into sub-queries..."}

            sources = _search_sources(query, req, provider)
            t1 = time.time()
            print(f"  [chat/sse] search done: {(t1-t0)*1000:.0f}ms  ({len(sources)} sources)")

            for s in sources:
                yield {
                    "event": "source",
                    "data": json.dumps({
                        "text": s.get("text", "")[:500],
                        "collection_display": s.get("collection_display", ""),
                        "episode_title": s.get("episode_title", ""),
                        "timestamp": s.get("timestamp", "00:00"),
                        "start_sec": s.get("start_sec", 0),
                        "end_sec": s.get("end_sec", 0),
                        "url": s.get("url", ""),
                    }),
                }

            db = get_database()
            session_id = _get_or_create_session(req, provider.name, req.model)
            db.add_message(session_id, "user", query)
            yield {"event": "session", "data": session_id}

            rag_messages = _build_rag_messages(req.messages, sources)
            full_response = ""
            first_token = True
            for chunk in provider.stream(rag_messages, model=req.model):
                if first_token:
                    print(f"  [chat/sse] first token: {(time.time()-t1)*1000:.0f}ms after search")
                    first_token = False
                full_response += chunk
                yield {"event": "token", "data": chunk}

            db.add_message(session_id, "assistant", full_response, sources=_sources_for_db(sources))
            yield {"event": "done", "data": session_id}
            print(f"  [chat/sse] TOTAL: {(time.time()-t0)*1000:.0f}ms")

        except Exception as e:
            print(f"  [chat/sse] ERROR: {e}")
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())
