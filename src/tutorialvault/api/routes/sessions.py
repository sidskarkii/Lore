"""Session management endpoints — list, get, rename, delete, search history."""

from fastapi import APIRouter, HTTPException

from ..schemas import (
    SessionInfo,
    SessionDetail,
    SessionsResponse,
    RenameSessionRequest,
    SearchMessagesRequest,
    SearchMessagesResponse,
    MessageInfo,
)
from ...core.database import get_database

router = APIRouter(tags=["sessions"])


@router.get(
    "/api/sessions",
    response_model=SessionsResponse,
    summary="List chat sessions",
    description="Returns all chat sessions, most recent first, with message counts.",
)
def list_sessions(limit: int = 50, offset: int = 0):
    db = get_database()
    sessions = db.list_sessions(limit=limit, offset=offset)
    return SessionsResponse(
        sessions=[
            SessionInfo(
                id=s["id"],
                title=s["title"],
                created_at=s["created_at"],
                updated_at=s["updated_at"],
                provider=s.get("provider"),
                model=s.get("model"),
                message_count=s.get("message_count", 0),
            )
            for s in sessions
        ]
    )


@router.get(
    "/api/sessions/{session_id}",
    response_model=SessionDetail,
    summary="Get a session with all messages",
    description="Returns the full session including all messages and their sources.",
)
def get_session(session_id: str):
    db = get_database()
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionDetail(
        id=session["id"],
        title=session["title"],
        created_at=session["created_at"],
        updated_at=session["updated_at"],
        provider=session.get("provider"),
        model=session.get("model"),
        messages=[
            MessageInfo(
                id=m["id"],
                role=m["role"],
                content=m["content"],
                sources=m.get("sources", []),
                created_at=m["created_at"],
            )
            for m in session["messages"]
        ],
    )


@router.patch(
    "/api/sessions/{session_id}",
    summary="Rename a session",
    description="Update the title of a chat session.",
)
def rename_session(session_id: str, req: RenameSessionRequest):
    db = get_database()
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    db.update_session_title(session_id, req.title)
    return {"status": "ok", "id": session_id, "title": req.title}


@router.delete(
    "/api/sessions/{session_id}",
    summary="Delete a session",
    description="Permanently deletes a session and all its messages.",
)
def delete_session(session_id: str):
    db = get_database()
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete_session(session_id)
    return {"status": "ok", "deleted": session_id}


@router.post(
    "/api/sessions/search",
    response_model=SearchMessagesResponse,
    summary="Search message history",
    description="Full-text search across all chat messages using FTS5.",
)
def search_messages(req: SearchMessagesRequest):
    db = get_database()
    results = db.search_messages(req.query, limit=req.limit)
    return SearchMessagesResponse(
        query=req.query,
        results=[
            MessageInfo(
                id=r["id"],
                role=r["role"],
                content=r["content"],
                sources=[],
                created_at=r["created_at"],
                session_id=r["session_id"],
                session_title=r.get("session_title"),
            )
            for r in results
        ],
    )
