"""Provider management endpoints — detect, install, switch, test."""

import time

from fastapi import APIRouter, HTTPException

from ..schemas import (
    ProvidersResponse,
    ProviderInfo,
    ProviderModelInfo,
    SetActiveRequest,
    InstallRequest,
    InstallResponse,
    TestConnectionRequest,
    TestConnectionResponse,
)
from ...providers.registry import get_registry

router = APIRouter(tags=["providers"])


@router.get(
    "/api/providers",
    response_model=ProvidersResponse,
    summary="List all providers with status",
    description=(
        "Detects which CLI tools are installed, their auth status, "
        "available models, and which is currently active. Used by the "
        "frontend settings page and setup wizard."
    ),
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
            user=info["user"],
            error=info["error"],
            models=[ProviderModelInfo(**m) for m in info["models"]],
            free_model_count=info["free_model_count"],
            install_command=info["install_command"],
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
    description="Sets which provider handles chat requests.",
)
def set_active(req: SetActiveRequest):
    registry = get_registry()
    try:
        registry.active = req.provider
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "active": req.provider}


@router.post(
    "/api/providers/install",
    response_model=InstallResponse,
    summary="Install a provider's CLI tool",
    description=(
        "Runs the provider's install command (e.g. npm install -g @kilocode/cli). "
        "Requires user permission — the frontend should confirm before calling this."
    ),
)
def install_provider(req: InstallRequest):
    registry = get_registry()
    provider = registry.get(req.provider)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {req.provider}")
    if not provider.install_command:
        return InstallResponse(
            success=False,
            provider=req.provider,
            error="This provider has no install command (configure it manually).",
        )

    success = registry.install(req.provider)
    return InstallResponse(
        success=success,
        provider=req.provider,
        error=None if success else "Install command failed. Check terminal for details.",
    )


@router.post(
    "/api/providers/authorize",
    summary="Authorize a provider to use existing credentials",
    description=(
        "For providers like Claude Code that need permission to copy "
        "credentials into an isolated config. The user's main setup is not modified."
    ),
)
def authorize_provider(req: InstallRequest):
    registry = get_registry()
    provider = registry.get(req.provider)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {req.provider}")

    if hasattr(provider, "authorize"):
        success = provider.authorize()
        return {
            "success": success,
            "provider": req.provider,
            "error": None if success else "Could not find credentials to copy.",
        }
    return {"success": True, "provider": req.provider, "error": None}


@router.post(
    "/api/providers/test",
    response_model=TestConnectionResponse,
    summary="Test a provider connection",
    description=(
        "Sends a simple test message to verify the provider works. "
        "Returns latency and a preview of the response."
    ),
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
            error="Provider CLI not installed",
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
