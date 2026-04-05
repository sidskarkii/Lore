"""Lore server — run with: python -m lore"""

import threading
import uvicorn

from .api.app import create_app

app = create_app()


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


def main(host: str = "127.0.0.1", port: int = 8000):
    print(f"\n  Lore API starting on http://{host}:{port}")
    print(f"  Docs: http://{host}:{port}/api/docs")
    print(f"  Health: http://{host}:{port}/api/health\n")
    threading.Thread(target=_warmup, daemon=True).start()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
