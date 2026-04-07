"""Search endpoint — hybrid retrieval from the knowledge base."""

from fastapi import APIRouter, HTTPException

from ..schemas import SearchRequest, SearchResponse, SearchResult
from ...core.search import get_search_engine

router = APIRouter(tags=["search"])


@router.post(
    "/api/search",
    response_model=SearchResponse,
    summary="Search the knowledge base",
    description=(
        "Runs hybrid search (vector + BM25) with RRF fusion and cross-encoder "
        "reranking. Returns chunks with source metadata and timestamps."
    ),
)
def search(req: SearchRequest):
    try:
        engine = get_search_engine()
        results = engine.search(
            query=req.query,
            n_results=req.n_results,
            topic=req.topic,
            subtopic=req.subtopic,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    return SearchResponse(
        query=req.query,
        total=len(results),
        results=[
            SearchResult(
                text=r.get("text", ""),
                collection=r.get("collection", ""),
                collection_display=r.get("collection_display", ""),
                episode_num=r.get("episode_num", 0),
                episode_title=r.get("episode_title", ""),
                timestamp=r.get("timestamp", "00:00"),
                start_sec=r.get("start_sec", 0),
                end_sec=r.get("end_sec", 0),
                url=r.get("url", ""),
                topic=r.get("topic", ""),
                subtopic=r.get("subtopic", ""),
            )
            for r in results
        ],
    )
