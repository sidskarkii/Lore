"""Collection management endpoints — list and delete indexed content."""

from fastapi import APIRouter, HTTPException

from ..schemas import (
    CollectionsResponse,
    CollectionInfo,
    EpisodeInfo,
    DeleteCollectionRequest,
)
from ...core.store import Store

router = APIRouter(tags=["collections"])


@router.get(
    "/api/collections",
    response_model=CollectionsResponse,
    summary="List all indexed collections",
    description=(
        "Returns all collections in the knowledge base with episode counts "
        "and metadata. Used by the Collections tab in the frontend."
    ),
)
def list_collections():
    store = Store()

    try:
        collections = store.list_collections()
        total = store.chunk_count()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {e}")

    return CollectionsResponse(
        total_chunks=total,
        collections=[
            CollectionInfo(
                collection=c["collection"],
                collection_display=c["collection_display"],
                topic=c["topic"],
                subtopic=c["subtopic"],
                episode_count=c["episode_count"],
                episodes=[
                    EpisodeInfo(
                        episode_num=ep["episode_num"],
                        episode_title=ep["episode_title"],
                    )
                    for ep in c["episodes"]
                ],
            )
            for c in collections
        ],
    )


@router.delete(
    "/api/collections",
    summary="Delete a collection",
    description="Removes all chunks for a collection from the knowledge base.",
)
def delete_collection(req: DeleteCollectionRequest):
    store = Store()

    try:
        store.delete_collection(req.collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}")

    return {"status": "ok", "deleted": req.collection}
