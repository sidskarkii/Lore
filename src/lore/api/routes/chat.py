"""Chat endpoint — RAG-grounded conversation with streaming support.

POST /api/chat         — full response (non-streaming)
POST /api/chat/stream  — SSE streaming (preferred)
WS   /api/chat/ws      — WebSocket streaming (legacy/desktop)
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..schemas import ChatRequest, ChatResponse, ChatMessage, SearchResult
from ...core.database import get_database
from ...core.search import SearchEngine
from ...providers.registry import get_registry


def _get_or_create_session(req: ChatRequest, provider_name: str, model: str | None) -> str:
    """Resume existing session or create a new one. Returns session_id."""
    db = get_database()
    if req.session_id:
        session = db.get_session(req.session_id)
        if session:
            return req.session_id
    # Create new session — title from first user message
    title = "New Chat"
    for m in req.messages:
        if m.role == "user":
            title = m.content[:80]
            break
    session = db.create_session(title=title, provider=provider_name, model=model)
    return session["id"]


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

router = APIRouter(tags=["chat"])

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant with access to a curated library of tutorials "
    "and documentation. Answer the user's question based on the provided sources. "
    "Be specific — mention exact tool names, menu paths, shortcuts, and node names "
    "when they appear in the sources.\n\n"
    "At the end of your answer, list the sources with their timestamps so the user "
    "can jump directly to that moment in the video."
)


def _build_rag_messages(
    messages: list[ChatMessage],
    sources: list[dict],
) -> list[dict]:
    """Build the message list with RAG context injected."""
    # Format sources into context block
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

    # Build final messages
    result = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (all but the last user message)
    for m in messages[:-1]:
        result.append({"role": m.role, "content": m.content})

    # Last user message gets the RAG context appended
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

    registry = get_registry()

    # Override active provider if specified
    if req.provider:
        provider = registry.get(req.provider)
        if not provider:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {req.provider}")
    else:
        provider = registry.active
        if not provider:
            raise HTTPException(status_code=400, detail="No active provider configured")

    # Search for relevant sources
    last_user_msg = ""
    for m in reversed(req.messages):
        if m.role == "user":
            last_user_msg = m.content
            break

    try:
        engine = SearchEngine()
        if req.multi_hop:
            try:
                sources = engine.search_multi_hop(
                    last_user_msg, provider=provider,
                    n_results=req.n_sources, topic=req.topic, subtopic=req.subtopic,
                )
            except Exception:
                sources = engine.search(last_user_msg, n_results=req.n_sources, topic=req.topic, subtopic=req.subtopic)
        else:
            sources = engine.search(last_user_msg, n_results=req.n_sources, topic=req.topic, subtopic=req.subtopic)
    except Exception:
        sources = []

    # Build RAG-augmented messages and send to provider
    rag_messages = _build_rag_messages(req.messages, sources)

    try:
        answer = provider.chat(rag_messages, model=req.model)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Provider error: {e}")

    # Persist to SQLite
    db = get_database()
    session_id = _get_or_create_session(req, provider.name, req.model)
    db.add_message(session_id, "user", last_user_msg)
    db.add_message(session_id, "assistant", answer, sources=_sources_for_db(sources))

    source_results = [
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

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=source_results,
        provider=provider.name,
        model=req.model or "default",
    )


@router.websocket("/api/chat/ws")
async def chat_stream_ws(ws: WebSocket):
    """WebSocket endpoint for streaming chat responses.

    Client sends a ChatRequest JSON, server streams back:
      {"type": "source", "data": SearchResult}     — for each retrieved source
      {"type": "token",  "data": "text chunk"}      — for each response token
      {"type": "done",   "data": null}               — when complete
      {"type": "error",  "data": "error message"}    — on failure
    """
    await ws.accept()

    try:
        raw = await ws.receive_text()
        req = ChatRequest.model_validate_json(raw)

        registry = get_registry()
        provider = registry.get(req.provider) if req.provider else registry.active
        if not provider:
            await ws.send_text(json.dumps({"type": "error", "data": "No active provider"}))
            await ws.close()
            return

        # Search for sources
        last_user_msg = ""
        for m in reversed(req.messages):
            if m.role == "user":
                last_user_msg = m.content
                break

        try:
            engine = SearchEngine()
            if req.multi_hop:
                await ws.send_text(json.dumps({"type": "status", "data": "Decomposing query into sub-queries..."}))
                try:
                    sources = engine.search_multi_hop(
                        last_user_msg, provider=provider,
                        n_results=req.n_sources, topic=req.topic, subtopic=req.subtopic,
                    )
                except Exception:
                    sources = engine.search(last_user_msg, n_results=req.n_sources, topic=req.topic, subtopic=req.subtopic)
            else:
                sources = engine.search(last_user_msg, n_results=req.n_sources, topic=req.topic, subtopic=req.subtopic)
        except Exception:
            sources = []

        # Send sources to client first
        for s in sources:
            await ws.send_text(json.dumps({
                "type": "source",
                "data": {
                    "text": s.get("text", "")[:500],
                    "collection_display": s.get("collection_display", ""),
                    "episode_title": s.get("episode_title", ""),
                    "timestamp": s.get("timestamp", "00:00"),
                    "start_sec": s.get("start_sec", 0),
                    "end_sec": s.get("end_sec", 0),
                    "url": s.get("url", ""),
                },
            }))

        # Persist user message + create/resume session
        db = get_database()
        session_id = _get_or_create_session(req, provider.name, req.model)
        db.add_message(session_id, "user", last_user_msg)

        await ws.send_text(json.dumps({"type": "session", "data": session_id}))

        # Stream LLM response
        rag_messages = _build_rag_messages(req.messages, sources)
        full_response = ""
        for chunk in provider.stream(rag_messages, model=req.model):
            full_response += chunk
            await ws.send_text(json.dumps({"type": "token", "data": chunk}))

        # Persist assistant response
        db.add_message(session_id, "assistant", full_response, sources=_sources_for_db(sources))

        await ws.send_text(json.dumps({"type": "done", "data": session_id}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "data": str(e)}))
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@router.post(
    "/api/chat/stream",
    summary="Stream chat response via SSE",
    description=(
        "Same pipeline as /api/chat but returns a Server-Sent Events stream. "
        "Events: source (retrieved sources), session (session ID), token "
        "(response chunks), done (completion). Preferred over WebSocket for "
        "unidirectional streaming — simpler clients, auto-reconnect, HTTP/2 "
        "multiplexing."
    ),
)
async def chat_stream_sse(req: ChatRequest):
    from sse_starlette.sse import EventSourceResponse

    registry = get_registry()
    provider = registry.get(req.provider) if req.provider else registry.active
    if not provider:
        return EventSourceResponse(
            iter([{"event": "error", "data": "No active provider"}])
        )

    # Extract last user message
    last_user_msg = ""
    for m in reversed(req.messages):
        if m.role == "user":
            last_user_msg = m.content
            break

    async def event_generator():
        t0 = time.time()
        try:
            # Search
            engine = SearchEngine()
            print(f"  [chat/sse] query: {last_user_msg[:80]!r}")
            if req.multi_hop:
                yield {"event": "status", "data": "Decomposing query into sub-queries..."}
                try:
                    sources = engine.search_multi_hop(
                        last_user_msg, provider=provider,
                        n_results=req.n_sources, topic=req.topic, subtopic=req.subtopic,
                    )
                except Exception:
                    sources = engine.search(last_user_msg, n_results=req.n_sources, topic=req.topic, subtopic=req.subtopic)
            else:
                sources = engine.search(last_user_msg, n_results=req.n_sources, topic=req.topic, subtopic=req.subtopic)

            t1 = time.time()
            print(f"  [chat/sse] search done: {(t1-t0)*1000:.0f}ms  ({len(sources)} sources)")

            # Send sources
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

            # Create/resume session
            db = get_database()
            session_id = _get_or_create_session(req, provider.name, req.model)
            db.add_message(session_id, "user", last_user_msg)
            yield {"event": "session", "data": session_id}

            # Stream LLM response
            rag_messages = _build_rag_messages(req.messages, sources)
            full_response = ""
            first_token = True
            for chunk in provider.stream(rag_messages, model=req.model):
                if first_token:
                    print(f"  [chat/sse] first token: {(time.time()-t1)*1000:.0f}ms after search")
                    first_token = False
                full_response += chunk
                yield {"event": "token", "data": chunk}

            # Persist
            db.add_message(session_id, "assistant", full_response, sources=_sources_for_db(sources))
            yield {"event": "done", "data": session_id}
            print(f"  [chat/sse] TOTAL: {(time.time()-t0)*1000:.0f}ms")

        except Exception as e:
            print(f"  [chat/sse] ERROR: {e}")
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())
