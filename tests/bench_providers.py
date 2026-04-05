"""Benchmark all providers on a real RAG query against the live index."""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

QUERY = "What are the most important things to know as a beginner?"

def bench():
    from lore.core.search import SearchEngine
    from lore.providers.registry import get_registry

    registry = get_registry()
    all_providers = list(registry._providers.values())

    if not all_providers:
        print("No providers registered.")
        return

    # Run search once (shared for all providers — not what we're benchmarking)
    print(f"Query: {QUERY!r}\n")
    print("Warming up search pipeline...")
    engine = SearchEngine()
    t0 = time.perf_counter()
    sources = engine.search(QUERY, n_results=3)
    search_ms = (time.perf_counter() - t0) * 1000
    print(f"Search: {search_ms:.0f}ms — {len(sources)} sources\n")

    if not sources:
        print("No chunks indexed yet — ingest something first.")
        return

    # Build RAG prompt once
    context = "\n\n---\n\n".join(
        f"[{i+1}] {s.get('episode_title','')} @ {s.get('timestamp','')}\n{s.get('text','')[:400]}"
        for i, s in enumerate(sources)
    )
    messages = [
        {"role": "system", "content": "Answer based on the sources provided."},
        {"role": "user", "content": f"{QUERY}\n\nSources:\n{context}"},
    ]

    print(f"{'Provider':<16} {'Status':<10} {'First token':>12} {'Total':>10} {'Response preview'}")
    print("-" * 90)

    results = []
    for p in all_providers:
        name = p.name
        try:
            status = p.status()
            installed = getattr(status, 'installed', False)
            authenticated = getattr(status, 'authenticated', False)
            if not installed or not authenticated:
                print(f"  {name:<14} SKIP       {'—':>12} {'—':>10}  not installed/auth'd")
                continue
        except Exception as e:
            print(f"  {name:<14} ERROR      {'—':>12} {'—':>10}  status() failed: {e}")
            continue

        try:
            t_start = time.perf_counter()
            first_token_ms = None
            chunks = []

            for chunk in p.stream(messages):
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - t_start) * 1000
                chunks.append(chunk)

            total_ms = (time.perf_counter() - t_start) * 1000
            response = "".join(chunks)
            preview = response.replace("\n", " ")[:60]

            print(f"  {name:<14} {'PASS':<10} {first_token_ms:>10.0f}ms {total_ms:>8.0f}ms  {preview}…")
            results.append((name, first_token_ms, total_ms))

        except Exception as e:
            print(f"  {name:<14} {'FAIL':<10} {'—':>12} {'—':>10}  {str(e)[:60]}")

    if results:
        print("\n--- Ranking by first token ---")
        for name, ft, tot in sorted(results, key=lambda x: x[1]):
            print(f"  {name:<16} first={ft:.0f}ms  total={tot:.0f}ms")

if __name__ == "__main__":
    bench()
