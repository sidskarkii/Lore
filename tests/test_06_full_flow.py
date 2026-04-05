"""Full end-to-end flow test.

Ingests real content → searches it → chats about it via provider → persists session.
This is the complete pipeline that the frontend will use.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

COURSE_DIR = "D:/Courses/VFXGRACE/Blender Animal Full Tutorial/Vfx Grace - Blender Animal Full Tutorial/Vfx Grace - Blender Animal Full Tutorial/1 - Blender Creature Effects The Complete WorkFlow En"


def test_full_flow():
    # ── 1. Ingest 3 episodes via SRT (fast, no transcription) ─────────
    print("=" * 60)
    print("  1. INGEST — 3 episodes from course SRTs")
    print("=" * 60)

    from lore.core.ingest import Ingester

    ing = Ingester()

    srts = [
        (1, "Introduction", f"{COURSE_DIR}/01 Introduction.srt"),
        (2, "Sculpting Tools", f"{COURSE_DIR}/02 Software Basics - Sculpting Tools.srt"),
        (3, "Mesh Editing Tools", f"{COURSE_DIR}/03 Software Basics - Mesh Editing Tools.srt"),
    ]

    total_chunks = 0
    for ep_num, title, srt_path in srts:
        if not os.path.exists(srt_path):
            print(f"  SKIP: {srt_path} not found")
            continue
        n = ing.ingest_srt(
            srt_path, name="Wolf Modeling Course", topic="3d", subtopic="blender",
            episode_num=ep_num, episode_title=title,
            url=f"file://{srt_path}",
        )
        print(f"  ep{ep_num:02d} {title}: {n} chunks")
        total_chunks += n

    assert total_chunks > 0, "No chunks ingested"
    print(f"\n  Total: {total_chunks} chunks ingested")

    # ── 2. List collections ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  2. COLLECTIONS")
    print("=" * 60)

    collections = ing.store.list_collections()
    wolf_collection = None
    for c in collections:
        if "Wolf" in c["collection_display"]:
            wolf_collection = c
            print(f"  {c['collection_display']}: {c['episode_count']} episodes")
            for ep in c["episodes"]:
                print(f"    ep{ep['episode_num']:02d}: {ep['episode_title']}")

    assert wolf_collection is not None, "Wolf collection not found"

    # ── 3. Search ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  3. SEARCH — 3 queries")
    print("=" * 60)

    from lore.core.search import SearchEngine
    engine = SearchEngine(ing.store)

    queries = [
        "what sculpting brushes are available",
        "how to use mesh editing tools",
        "difficulties beginners face when learning modeling",
    ]

    for q in queries:
        print(f"\n  Q: \"{q}\"")
        t0 = time.time()
        results = engine.search(q, n_results=2, subtopic="blender")
        elapsed = time.time() - t0
        assert len(results) > 0, f"No results for: {q}"
        for r in results:
            print(f"    [{r['timestamp']}] ep{r['episode_num']:02d}: {r['text'][:100]}...")
        print(f"    ({elapsed:.1f}s)")

    # ── 4. Chat via provider ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  4. CHAT — ask a question via active provider")
    print("=" * 60)

    from lore.providers.registry import get_registry
    registry = get_registry()
    provider = registry.active
    print(f"  Provider: {provider.name if provider else 'NONE'}")

    if provider and provider.detect():
        # Build RAG context
        query = "What are the main sculpting brushes in Blender and what do they do?"
        print(f"  Q: \"{query}\"")

        results = engine.search(query, n_results=3)
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}] {r['collection_display']} ep{r['episode_num']:02d}: {r['episode_title']}\n"
                f"@ {r['timestamp']}\n{r['text'][:500]}"
            )
        context = "\n\n---\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": "Answer based on the tutorial sources provided. Be specific."},
            {"role": "user", "content": f"{query}\n\nSources:\n{context}"},
        ]

        print(f"  Sending to {provider.name}...")
        t0 = time.time()
        answer = provider.chat(messages)
        elapsed = time.time() - t0
        print(f"  Answer ({elapsed:.1f}s, {len(answer)} chars):")
        print(f"    {answer[:300]}...")
        assert len(answer) > 20, "Answer too short"
    else:
        print("  SKIP: no provider available")

    # ── 5. Persist to SQLite ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  5. PERSIST — save chat to SQLite")
    print("=" * 60)

    from lore.core.database import Database
    import tempfile

    db = Database(os.path.join(tempfile.gettempdir(), "tv_test_flow.db"))

    session = db.create_session(
        title="Sculpting Brushes Question",
        provider=provider.name if provider else "none",
    )
    print(f"  Session: {session['id']}")

    db.add_message(session["id"], "user", "What are the main sculpting brushes?")
    db.add_message(session["id"], "assistant", answer if provider and provider.detect() else "Test answer",
                   sources=[{"timestamp": r["timestamp"], "episode_title": r["episode_title"]} for r in results[:3]])

    # Verify
    full = db.get_session(session["id"])
    assert len(full["messages"]) == 2
    assert full["messages"][1]["sources"]
    print(f"  Messages: {len(full['messages'])}")
    print(f"  Sources on answer: {len(full['messages'][1]['sources'])}")

    # Search history
    found = db.search_messages("sculpting brushes")
    assert len(found) > 0
    print(f"  History search 'sculpting brushes': {len(found)} result(s)")

    db.close()
    os.unlink(os.path.join(tempfile.gettempdir(), "tv_test_flow.db"))

    # ── 6. API server check ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  6. API — verify all routes compile")
    print("=" * 60)

    from lore.api.app import create_app
    app = create_app()
    routes = [r.path for r in app.routes if hasattr(r, "path") and r.path.startswith("/api")]
    print(f"  {len(routes)} API routes registered")
    for r in sorted(routes):
        print(f"    {r}")

    print(f"\n{'=' * 60}")
    print("  FULL FLOW TEST PASSED")
    print("=" * 60)
    print(f"\n  Pipeline: SRT -> chunk -> embed -> store -> search -> chat -> persist")
    print(f"  Chunks: {total_chunks}")
    print(f"  Collections: {len(collections)}")
    print(f"  Provider: {provider.name if provider else 'none'}")
    print(f"  API routes: {len(routes)}")


if __name__ == "__main__":
    test_full_flow()
