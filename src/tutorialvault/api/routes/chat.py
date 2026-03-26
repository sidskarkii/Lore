"""Chat endpoint — RAG-grounded conversation with streaming support.

POST /api/chat        — full response (non-streaming)
WS   /api/chat/stream — WebSocket streaming
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..schemas import ChatRequest, ChatResponse, ChatMessage, SearchResult
from ...core.search import SearchEngine
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
        sources = engine.search(last_user_msg, n_results=req.n_sources, topic=req.topic, subtopic=req.subtopic)
    except Exception:
        sources = []

    # Build RAG-augmented messages and send to provider
    rag_messages = _build_rag_messages(req.messages, sources)

    try:
        answer = provider.chat(rag_messages, model=req.model)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Provider error: {e}")

    return ChatResponse(
        answer=answer,
        sources=[
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
        ],
        provider=provider.name,
        model=req.model or "default",
    )


@router.websocket("/api/chat/stream")
async def chat_stream(ws: WebSocket):
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

        # Stream LLM response
        rag_messages = _build_rag_messages(req.messages, sources)
        for chunk in provider.stream(rag_messages, model=req.model):
            await ws.send_text(json.dumps({"type": "token", "data": chunk}))

        await ws.send_text(json.dumps({"type": "done", "data": None}))

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
