"""Benchmark: Manual hybrid search vs LanceDB native hybrid search.

Compares:
  A) Current approach: separate vector_search + fts_search + manual RRF
  B) LanceDB native: query_type="hybrid" with built-in RRFReranker

Measures: speed, result overlap, result quality (FlashRank scores).
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

COURSE_DIR = "D:/Courses/VFXGRACE/Blender Animal Full Tutorial/Vfx Grace - Blender Animal Full Tutorial/Vfx Grace - Blender Animal Full Tutorial/1 - Blender Creature Effects The Complete WorkFlow En"

QUERIES = [
    "what sculpting brushes are available",
    "how to use mesh editing tools",
    "difficulties beginners face when learning modeling",
    "Compare sculpting workflow to mesh editing approach",
    "how to create detailed wrinkles on a character face",
]


def bench():
    # ── 1. Ingest ─────────────────────────────────────────────────────
    print("=" * 70)
    print("  SETUP: Ingest 3 episodes")
    print("=" * 70)

    from lore.core.ingest import Ingester
    from lore.core.search import SearchEngine, _rerank_with_scores, _rrf, _rerank
    from lore.core.embed import embed_texts
    from lore.core.store import Store

    ing = Ingester()
    srts = [
        (1, "Introduction", f"{COURSE_DIR}/01 Introduction.srt"),
        (2, "Sculpting Tools", f"{COURSE_DIR}/02 Software Basics - Sculpting Tools.srt"),
        (3, "Mesh Editing Tools", f"{COURSE_DIR}/03 Software Basics - Mesh Editing Tools.srt"),
    ]
    for ep_num, title, srt_path in srts:
        if not os.path.exists(srt_path):
            print(f"  SKIP: {srt_path}")
            continue
        ing.ingest_srt(
            srt_path, name="Wolf Modeling Course", topic="3d", subtopic="blender",
            episode_num=ep_num, episode_title=title,
        )
    print(f"  {ing.store.chunk_count()} chunks ready\n")

    store = ing.store
    engine = SearchEngine(store)

    # ── 2. Benchmark: Manual hybrid (current) ─────────────────────────
    print("=" * 70)
    print("  A) MANUAL HYBRID (current: vector + fts + RRF)")
    print("=" * 70)

    manual_results = {}
    manual_times = []

    for q in QUERIES:
        t0 = time.perf_counter()
        results = engine.search(q, n_results=5, subtopic="blender")
        elapsed = time.perf_counter() - t0
        manual_times.append(elapsed)
        manual_results[q] = results
        ids = [r["id"] for r in results]
        eps = sorted(set(r["episode_num"] for r in results))
        print(f"\n  Q: \"{q}\"")
        print(f"  Time: {elapsed*1000:.0f}ms | Results: {len(results)} | Episodes: {eps}")
        for r in results[:3]:
            print(f"    [{r['timestamp']}] ep{r['episode_num']:02d}: {r['text'][:70]}...")

    # ── 3. Benchmark: LanceDB native hybrid with pre-computed vector ──
    print(f"\n{'=' * 70}")
    print("  B) LANCEDB NATIVE HYBRID + PRE-EMBEDDED QUERY")
    print("=" * 70)

    from lancedb.rerankers import RRFReranker
    tbl = store._get_or_create_table()

    native_results = {}
    native_times = []

    for q in QUERIES:
        t0 = time.perf_counter()

        # Pre-embed query (same as manual path)
        query_vec = embed_texts([q])[0]

        try:
            raw = (
                tbl.search(query_type="hybrid")
                .vector(query_vec)
                .text(q)
                .limit(30)
                .where("subtopic = 'blender'", prefilter=True)
                .rerank(RRFReranker(K=60, return_score="all"))
                .to_pandas()
                .to_dict("records")
            )
        except Exception as e:
            print(f"\n  Q: \"{q}\"")
            print(f"  ERROR: {e}")
            native_results[q] = []
            native_times.append(0)
            continue

        reranked = _rerank(q, raw, 5)
        elapsed = time.perf_counter() - t0
        native_times.append(elapsed)
        native_results[q] = reranked
        ids = [r["id"] for r in reranked]
        eps = sorted(set(r.get("episode_num", 0) for r in reranked))
        print(f"\n  Q: \"{q}\"")
        print(f"  Time: {elapsed*1000:.0f}ms | Results: {len(reranked)} | Episodes: {eps}")
        for r in reranked[:3]:
            print(f"    [{r.get('timestamp', '?')}] ep{r.get('episode_num', 0):02d}: {r.get('text', '')[:70]}...")

    # ── 5. Comparison ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  COMPARISON")
    print("=" * 70)

    print(f"\n  {'Query':<55} {'Manual':>8} {'Native':>8}")
    print(f"  {'-'*55} {'-'*8} {'-'*8}")
    for i, q in enumerate(QUERIES):
        mt = manual_times[i] * 1000
        nt = native_times[i] * 1000
        print(f"  {q[:55]:<55} {mt:>6.0f}ms {nt:>6.0f}ms")

    avg_m = sum(manual_times) / len(manual_times) * 1000
    avg_n = sum(native_times) / len(native_times) * 1000
    print(f"\n  {'AVERAGE':<55} {avg_m:>6.0f}ms {avg_n:>6.0f}ms")

    # Result overlap
    print(f"\n  Result Overlap (Manual vs Native):")
    for q in QUERIES:
        m_ids = set(r["id"] for r in manual_results[q])
        n_ids = set(r.get("id", "") for r in native_results.get(q, []))
        overlap = len(m_ids & n_ids) / max(len(m_ids), 1) * 100
        print(f"    {q[:55]:<55} {overlap:.0f}%")

    print(f"\n  BENCHMARK COMPLETE")


if __name__ == "__main__":
    bench()
