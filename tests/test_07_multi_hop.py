"""Multi-hop search test.

Tests the query decomposition → multi-pass search → combine → rerank pipeline.
Uses a StubProvider for deterministic testing without requiring a live LLM.
Also tests with a real provider if one is available.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

COURSE_DIR = "D:/Courses/VFXGRACE/Blender Animal Full Tutorial/Vfx Grace - Blender Animal Full Tutorial/Vfx Grace - Blender Animal Full Tutorial/1 - Blender Creature Effects The Complete WorkFlow En"


class StubProvider:
    """Fake provider that returns hardcoded sub-queries for testing."""
    name = "stub"

    def detect(self):
        return True

    def chat(self, messages, model=None):
        # Extract query from the decomposition prompt
        content = messages[-1]["content"]
        if "sculpt" in content.lower() and "mesh" in content.lower():
            return (
                "sculpting brushes and tools in Blender\n"
                "mesh editing tools and techniques\n"
                "differences between sculpting and mesh editing workflows"
            )
        if "beginner" in content.lower() or "difficult" in content.lower():
            return (
                "common difficulties when learning 3D modeling\n"
                "beginner tips for Blender sculpting\n"
                "introduction to mesh editing for new users"
            )
        # Default: return the original query as-is (triggers single-hop fallback)
        return content.split("Question: ")[-1].strip()


def test_multi_hop():
    # ── 1. Ingest 3 episodes ──────────────────────────────────────────
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

    assert total_chunks > 0, "No chunks ingested — check SRT paths"
    print(f"\n  Total: {total_chunks} chunks ingested")

    # ── 2. Multi-hop search with StubProvider ─────────────────────────
    print(f"\n{'=' * 60}")
    print("  2. MULTI-HOP SEARCH — StubProvider")
    print("=" * 60)

    from lore.core.search import SearchEngine
    engine = SearchEngine(ing.store)
    stub = StubProvider()

    query = "Compare the sculpting workflow to the mesh editing approach and which is better for beginners"
    print(f"\n  Q: \"{query}\"")

    t0 = time.time()
    multi_results = engine.search_multi_hop(query, provider=stub, n_results=5, subtopic="blender")
    multi_time = time.time() - t0

    assert len(multi_results) > 0, "Multi-hop returned no results"
    multi_episodes = set(r["episode_num"] for r in multi_results)
    print(f"\n  Multi-hop results ({multi_time:.1f}s):")
    for r in multi_results:
        print(f"    [{r['timestamp']}] ep{r['episode_num']:02d}: {r['text'][:80]}...")
    print(f"  Episodes covered: {sorted(multi_episodes)}")

    # ── 3. Single-hop search (same query) ─────────────────────────────
    print(f"\n{'=' * 60}")
    print("  3. SINGLE-HOP SEARCH — same query")
    print("=" * 60)

    t0 = time.time()
    single_results = engine.search(query, n_results=5, subtopic="blender")
    single_time = time.time() - t0

    assert len(single_results) > 0, "Single-hop returned no results"
    single_episodes = set(r["episode_num"] for r in single_results)
    print(f"\n  Single-hop results ({single_time:.1f}s):")
    for r in single_results:
        print(f"    [{r['timestamp']}] ep{r['episode_num']:02d}: {r['text'][:80]}...")
    print(f"  Episodes covered: {sorted(single_episodes)}")

    # ── 4. Compare quality ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  4. COMPARE — multi-hop vs single-hop")
    print("=" * 60)

    print(f"  Multi-hop:  {len(multi_results)} results, {len(multi_episodes)} episodes: {sorted(multi_episodes)}")
    print(f"  Single-hop: {len(single_results)} results, {len(single_episodes)} episodes: {sorted(single_episodes)}")

    # Multi-hop should cover at least as many episodes (ideally more)
    # With 3 ingested episodes and a comparison query, multi-hop should hit 2+
    assert len(multi_episodes) >= 2, (
        f"Multi-hop only covered {len(multi_episodes)} episode(s) — expected at least 2 for a comparison query"
    )

    # Both should return same dict format
    for key in ["id", "text", "collection", "episode_num", "episode_title", "timestamp", "start_sec", "end_sec"]:
        assert key in multi_results[0], f"Missing key '{key}' in multi-hop result"
        assert key in single_results[0], f"Missing key '{key}' in single-hop result"

    # ── 5. Graceful degradation (provider=None) ───────────────────────
    print(f"\n{'=' * 60}")
    print("  5. GRACEFUL DEGRADATION — provider=None")
    print("=" * 60)

    fallback = engine.search_multi_hop(query, provider=None, n_results=3, subtopic="blender")
    assert len(fallback) > 0, "Fallback with provider=None returned no results"
    print(f"  Fallback returned {len(fallback)} results (fell back to single-hop)")

    # ── 6. Simple query → should stay single-hop ──────────────────────
    print(f"\n{'=' * 60}")
    print("  6. SIMPLE QUERY — should not decompose")
    print("=" * 60)

    simple_query = "what sculpting brushes are available"
    simple_results = engine.search_multi_hop(simple_query, provider=stub, n_results=3, subtopic="blender")
    assert len(simple_results) > 0, "Simple query multi-hop returned no results"
    print(f"  Simple query returned {len(simple_results)} results")

    # ── 7. Real provider (if available) ───────────────────────────────
    print(f"\n{'=' * 60}")
    print("  7. REAL PROVIDER — live decomposition")
    print("=" * 60)

    from lore.providers.registry import get_registry
    registry = get_registry()
    provider = registry.active

    if provider and provider.detect():
        print(f"  Provider: {provider.name}")
        t0 = time.time()
        real_results = engine.search_multi_hop(
            query, provider=provider, n_results=5, subtopic="blender",
        )
        elapsed = time.time() - t0
        real_episodes = set(r["episode_num"] for r in real_results)
        print(f"  Real multi-hop ({elapsed:.1f}s): {len(real_results)} results, episodes: {sorted(real_episodes)}")
        for r in real_results:
            print(f"    [{r['timestamp']}] ep{r['episode_num']:02d}: {r['text'][:80]}...")
    else:
        print("  SKIP: no live provider available")

    # ── Done ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  MULTI-HOP TEST PASSED")
    print("=" * 60)
    sub_q_count = len(stub.chat([{"role": "user", "content": "Question: " + query}]).splitlines())
    print(f"\n  Pipeline: decompose -> {sub_q_count} sub-queries -> search each -> RRF combine -> rerank -> threshold -> expand")
    print(f"  Chunks ingested: {total_chunks}")
    print(f"  Multi-hop episodes: {sorted(multi_episodes)}")
    print(f"  Single-hop episodes: {sorted(single_episodes)}")


if __name__ == "__main__":
    test_multi_hop()
