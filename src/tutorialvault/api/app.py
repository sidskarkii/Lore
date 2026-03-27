"""FastAPI application — ties together all route modules.

This is the entry point for the API server. It configures CORS
(for the Tauri frontend), registers all routers, and sets up
OpenAPI documentation.

Run with: python -m tutorialvault.server
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import chat, collections, health, ingest, providers, search, sessions


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="TutorialVault API",
        description=(
            "Backend API for TutorialVault — a universal knowledge base that turns "
            "videos, docs, and playlists into a searchable, chat-ready library.\n\n"
            "The API serves the Tauri/Svelte desktop frontend. All business logic "
            "lives in core/ and providers/ — these routes are thin wrappers."
        ),
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # CORS — allow the Tauri frontend (and localhost dev server)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "tauri://localhost",    # Tauri v2 webview origin
            "http://localhost",     # Dev
            "http://localhost:1420", # Vite dev server (Tauri default)
            "http://localhost:5173", # Vite default
            "http://127.0.0.1:1420",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register route modules
    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(chat.router)
    app.include_router(providers.router)
    app.include_router(collections.router)
    app.include_router(sessions.router)
    app.include_router(ingest.router)

    return app
