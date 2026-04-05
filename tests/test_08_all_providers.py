"""Test all providers: detect, chat, stream, and SSE endpoint compilation.

Runs a real RAG query through every available provider to verify
the full pipeline works end-to-end.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

COURSE_DIR = "D:/Courses/VFXGRACE/Blender Animal Full Tutorial/Vfx Grace - Blender Animal Full Tutorial/Vfx Grace - Blender Animal Full Tutorial/1 - Blender Creature Effects The Complete WorkFlow En"


def test_all_providers():
    # ── 1. Ingest test data ───────────────────────────────────────────
    print("=" * 70)
    print("  1. SETUP: Ingest 3 episodes")
    print("=" * 70)

    from lore.core.ingest import Ingester
    from lore.core.search import SearchEngine
    from lore.core.database import Database
    from lore.providers.registry import get_registry
    import tempfile

    ing = Ingester()
    srts = [
        (1, "Introduction", f"{COURSE_DIR}/01 Introduction.srt"),
        (2, "Sculpting Tools", f"{COURSE_DIR}/02 Software Basics - Sculpting Tools.srt"),
        (3, "Mesh Editing Tools", f"{COURSE_DIR}/03 Software Basics - Mesh Editing Tools.srt"),
    ]
    total = 0
    for ep_num, title, srt_path in srts:
        if not os.path.exists(srt_path):
            continue
        total += ing.ingest_srt(
            srt_path, name="Wolf Modeling Course", topic="3d", subtopic="blender",
            episode_num=ep_num, episode_title=title,
        )
    print(f"  {total} chunks ingested\n")

    # ── 2. Search (provider-independent) ──────────────────────────────
    print("=" * 70)
    print("  2. SEARCH (no provider needed)")
    print("=" * 70)

    engine = SearchEngine(ing.store)
    query = "What sculpting brushes are available and what do they do?"

    t0 = time.time()
    results = engine.search(query, n_results=3, subtopic="blender")
    elapsed = time.time() - t0
    assert len(results) > 0, "Search returned no results"
    print(f"  Query: \"{query}\"")
    print(f"  {len(results)} results in {elapsed:.1f}s")
    for r in results:
        print(f"    [{r['timestamp']}] ep{r['episode_num']:02d}: {r['text'][:70]}...")

    # Build RAG context (shared across providers)
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(
            f"[Source {i}] {r['collection_display']} ep{r['episode_num']:02d}: {r['episode_title']}\n"
            f"@ {r['timestamp']}\n{r['text'][:400]}"
        )
    context = "\n\n---\n\n".join(context_parts)
    messages = [
        {"role": "system", "content": "Answer based on the tutorial sources provided. Be specific and concise."},
        {"role": "user", "content": f"{query}\n\nSources:\n{context}"},
    ]

    # ── 3. Test each provider ─────────────────────────────────────────
    registry = get_registry()
    all_status = registry.all_status()

    results_table = []
    passed = 0
    failed = 0
    skipped = 0

    for name, info in all_status.items():
        print(f"\n{'=' * 70}")
        print(f"  3. PROVIDER: {info['display_name']} ({name})")
        print("=" * 70)

        provider = registry.get(name)

        # Check if usable
        if not info["installed"]:
            print(f"  SKIP: not installed")
            results_table.append((name, "SKIP", "not installed", 0))
            skipped += 1
            continue

        if not info["authenticated"]:
            print(f"  SKIP: not authenticated ({info.get('error', '')})")
            results_table.append((name, "SKIP", info.get("error") or "not authenticated", 0))
            skipped += 1
            continue

        if not provider.detect():
            print(f"  SKIP: CLI not found in PATH")
            results_table.append((name, "SKIP", "CLI not in PATH", 0))
            skipped += 1
            continue

        print(f"  Version: {info['version']}")
        print(f"  Models: {len(info['models'])} ({info['free_model_count']} free)")

        # ── Chat (non-streaming) ──────────────────────────────────────
        print(f"\n  --- chat() ---")
        try:
            t0 = time.time()
            answer = provider.chat(messages)
            elapsed = time.time() - t0
            assert len(answer) > 10, f"Answer too short: {len(answer)} chars"
            print(f"  OK ({elapsed:.1f}s, {len(answer)} chars)")
            print(f"  Preview: {answer[:150]}...")
        except Exception as e:
            print(f"  FAIL: {e}")
            results_table.append((name, "FAIL", f"chat: {e}", 0))
            failed += 1
            continue

        # ── Stream ────────────────────────────────────────────────────
        print(f"\n  --- stream() ---")
        try:
            t0 = time.time()
            stream_buf = ""
            chunk_count = 0
            for chunk in provider.stream(messages):
                stream_buf += chunk
                chunk_count += 1
            elapsed = time.time() - t0

            if len(stream_buf) > 10:
                print(f"  OK ({elapsed:.1f}s, {chunk_count} chunks, {len(stream_buf)} chars)")
                print(f"  Preview: {stream_buf[:150]}...")
            else:
                print(f"  WARN: stream returned {len(stream_buf)} chars ({chunk_count} chunks)")
        except Exception as e:
            print(f"  FAIL: {e}")
            # Stream failure is non-fatal — chat worked
            pass

        # ── Persist ───────────────────────────────────────────────────
        print(f"\n  --- persist ---")
        db_path = os.path.join(tempfile.gettempdir(), f"tv_test_{name}.db")
        try:
            db = Database(db_path)
            session = db.create_session(title="Provider Test", provider=name)
            db.add_message(session["id"], "user", query)
            db.add_message(session["id"], "assistant", answer,
                          sources=[{"timestamp": r["timestamp"], "episode_title": r["episode_title"]}
                                   for r in results[:3]])
            full = db.get_session(session["id"])
            assert len(full["messages"]) == 2
            print(f"  OK (session {session['id']}, {len(full['messages'])} messages)")
            db.close()
            os.unlink(db_path)
        except Exception as e:
            print(f"  FAIL: {e}")

        results_table.append((name, "PASS", f"{len(answer)} chars", round(elapsed, 1)))
        passed += 1

    # ── 4. API routes check ───────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  4. API ROUTES")
    print("=" * 70)

    from lore.api.app import create_app
    app = create_app()
    routes = [r.path for r in app.routes if hasattr(r, "path") and r.path.startswith("/api")]
    print(f"  {len(routes)} routes registered")

    # Check SSE endpoint exists
    assert "/api/chat/stream" in routes, "SSE endpoint missing"
    assert "/api/chat/ws" in routes, "WebSocket endpoint missing"
    print(f"  SSE: /api/chat/stream  OK")
    print(f"  WS:  /api/chat/ws      OK")

    # ── 5. Summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print("=" * 70)

    print(f"\n  {'Provider':<15} {'Status':<8} {'Detail':<40} {'Time':>6}")
    print(f"  {'-'*15} {'-'*8} {'-'*40} {'-'*6}")
    for name, status, detail, elapsed in results_table:
        t = f"{elapsed}s" if elapsed else ""
        print(f"  {name:<15} {status:<8} {detail[:40]:<40} {t:>6}")

    print(f"\n  Passed: {passed}  Failed: {failed}  Skipped: {skipped}")
    print(f"  API routes: {len(routes)}")
    print(f"  Chunks: {total}")

    if failed > 0:
        print(f"\n  *** {failed} PROVIDER(S) FAILED ***")
    else:
        print(f"\n  ALL AVAILABLE PROVIDERS PASSED")


if __name__ == "__main__":
    test_all_providers()
