"""FastAPI application — ties together all route modules.

This is the entry point for the API server. It configures CORS
(for the Tauri frontend), registers all routers, and sets up
OpenAPI documentation.

Run with: python -m lore.server
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.routing import Mount

from .routes import chat, collections, health, ingest, providers, search, sessions
from ..mcp import create_mcp_server


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    mcp_server = create_mcp_server()
    mcp_app = mcp_server.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp_server.session_manager.run():
            yield

    app = FastAPI(
        title="Lore API",
        description=(
            "Backend API for Lore — a universal knowledge base that turns "
            "videos, docs, and playlists into a searchable, chat-ready library.\n\n"
            "The API serves the Tauri/React desktop frontend. All business logic "
            "lives in core/ and providers/ — these routes are thin wrappers."
        ),
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^(tauri://localhost|http://(localhost|127\.0\.0\.1)(:\d+)?)$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(chat.router)
    app.include_router(providers.router)
    app.include_router(collections.router)
    app.include_router(sessions.router)
    app.include_router(ingest.router)

    app.routes.append(Mount("/mcp", app=mcp_app))

    return app
