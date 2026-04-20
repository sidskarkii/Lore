"""Lore server — run with: python -m lore"""

import threading
import uvicorn

from .api.app import create_app

#builds  the FastAPI application (all the routes, CORS config, etc.)
app = create_app()

#pre-loads ONNX models by running a dummy embedding (embed_texts(["warmup"])) and initializing the reranker. If either fails, it just prints a warning and moves on rather than crashing the whole server.
def _warmup():
    """Load all models in background so first query is instant."""
    try:
        from .core.embed import embed_texts
        print("  [warmup] Loading embedding model…")
        embed_texts(["warmup"])
        print("  [warmup] Embedding model ready.")
    except Exception as e:
        print(f"  [warmup] Embedding failed: {e}")

    try:
        from .core.search import _get_ranker
        print("  [warmup] Loading reranker…")
        _get_ranker()
        print("  [warmup] Reranker ready.")
    except Exception as e:
        print(f"  [warmup] Reranker failed: {e}")

# Prints the URLs, then fires off _warmup on a background daemon thread so the server doesn't block waiting for models to load. Then starts uvicorn, which is the ASGI server that actually listens for HTTP requests and hands them to FastAPI.
def main():
    import sys

    if "--mcp-stdio" in sys.argv:
        from .mcp import create_mcp_server
        mcp = create_mcp_server()
        mcp.run(transport="stdio")
        return

    from .core.config import get_config
    cfg = get_config()
    host = cfg.get("server.host", "127.0.0.1")
    port = cfg.get("server.port", 8000)
    print(f"\n  Lore API starting on http://{host}:{port}")
    print(f"  Docs: http://{host}:{port}/api/docs")
    print(f"  Health: http://{host}:{port}/api/health")
    print(f"  MCP:  http://{host}:{port}/mcp\n")
    threading.Thread(target=_warmup, daemon=True).start()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
