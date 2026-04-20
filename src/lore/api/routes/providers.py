"""Provider management endpoints — list, switch, and test."""

import time

from fastapi import APIRouter, HTTPException

from ..schemas import (
    ProvidersResponse,
    ProviderInfo,
    ProviderModelInfo,
    SetActiveRequest,
    TestConnectionRequest,
    TestConnectionResponse,
)
from ...providers.registry import get_registry

router = APIRouter(tags=["providers"])


@router.get(
    "/api/providers",
    response_model=ProvidersResponse,
    summary="List providers with status",
)
def list_providers():
    registry = get_registry()
    all_status = registry.all_status()

    providers = [
        ProviderInfo(
            name=info["name"],
            display_name=info["display_name"],
            installed=info["installed"],
            authenticated=info["authenticated"],
            version=info["version"],
            error=info["error"],
            models=[ProviderModelInfo(**m) for m in info["models"]],
            free_model_count=info["free_model_count"],
            is_active=info["is_active"],
        )
        for info in all_status.values()
    ]

    active = registry.active
    return ProvidersResponse(
        providers=providers,
        active=active.name if active else None,
    )


@router.post(
    "/api/providers/active",
    summary="Switch the active provider",
)
def set_active(req: SetActiveRequest):
    registry = get_registry()
    try:
        registry.active = req.provider
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "active": req.provider}


@router.post(
    "/api/providers/test",
    response_model=TestConnectionResponse,
    summary="Test a provider connection",
)
def test_connection(req: TestConnectionRequest):
    registry = get_registry()

    if req.provider:
        provider = registry.get(req.provider)
        if not provider:
            raise HTTPException(status_code=404, detail=f"Unknown provider: {req.provider}")
    else:
        provider = registry.active
        if not provider:
            raise HTTPException(status_code=400, detail="No active provider")

    if not provider.detect():
        return TestConnectionResponse(
            success=False,
            provider=provider.name,
            model=req.model or "default",
            error="Provider not configured",
        )

    test_messages = [{"role": "user", "content": "Say 'hello' in one word."}]

    try:
        t0 = time.time()
        response = provider.chat(test_messages, model=req.model)
        latency = (time.time() - t0) * 1000

        return TestConnectionResponse(
            success=True,
            provider=provider.name,
            model=req.model or "default",
            latency_ms=round(latency),
            response_preview=response[:200],
        )
    except Exception as e:
        return TestConnectionResponse(
            success=False,
            provider=provider.name,
            model=req.model or "default",
            error=str(e),
        )
