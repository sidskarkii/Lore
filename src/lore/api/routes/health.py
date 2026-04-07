"""Health check endpoint."""

from fastapi import APIRouter

from ..schemas import HealthResponse
from ...core.config import get_config
from ...core.store import get_store
from ...providers.registry import get_registry

router = APIRouter(tags=["health"])


@router.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Server health check",
    description="Returns server status, model info, and basic stats.",
)
def health():
    cfg = get_config()
    registry = get_registry()
    active = registry.active

    try:
        total = get_store().chunk_count()
    except Exception:
        total = 0

    return HealthResponse(
        status="ok",
        version="0.1.0",
        embedding_model=cfg.get("embedding.model", ""),
        reranker_model=cfg.get("search.reranker_model", ""),
        total_chunks=total,
        active_provider=active.name if active else None,
    )
