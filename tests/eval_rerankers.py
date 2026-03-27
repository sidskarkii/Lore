"""Eval rerankers on REAL data from old TutorialVault LanceDB (211K chunks).

Strategy:
- Use BM25 (FTS) to get noisy initial candidates from the real corpus
- Run each reranker on those candidates
- Print ranked results side-by-side for manual quality judgment
- Also measure speed

Rerankers tested:
- FlashRank TinyBERT (~4MB)
- FlashRank MiniLM-L12 (~34MB)
- FlashRank MultiBERT (~150MB, multilingual)
- FlashRank rank-T5-flan (~110MB, zero-shot)
- No reranker (BM25 order as baseline)
"""

import time
import lancedb

OLD_DB = "F:/TutorialVault/.lancedb"

# Real queries a user would ask about Blender/Houdini tutorials
QUERIES = [
    "how to rig a character skeleton in blender",
    "weight painting bone influence",
    "houdini vop noise pattern",
    "sculpting detailed wrinkles on a face",
    "how to retopologize a high poly mesh",
    "particle system for hair and fur",
    "shader nodes for realistic skin material",
    "animation keyframe graph editor curves",
]

N_CANDIDATES = 20  # BM25 pulls this many
N_SHOW = 5  # Show top N after reranking

RERANKER_MODELS = [
    ("TinyBERT (4MB)", "ms-marco-TinyBERT-L-2-v2"),
    ("MiniLM-L12 (34MB)", "ms-marco-MiniLM-L-12-v2"),
    ("MultiBERT (150MB)", "ms-marco-MultiBERT-L-12"),
    ("rank-T5-flan (110MB)", "rank-T5-flan"),
]


def get_bm25_candidates(query: str, n: int = 20) -> list[dict]:
    """Pull candidates from old LanceDB via FTS."""
    db = lancedb.connect(OLD_DB)
    tbl = db.open_table("tutorials")
    try:
        results = tbl.search(query, query_type="fts").limit(n).to_pandas()
        return results.to_dict("records")
    except Exception as e:
        print(f"  FTS error: {e}")
        return []


def eval_reranker(model_name: str, display_name: str):
    """Load a FlashRank reranker and score all queries."""
    from flashrank import Ranker, RerankRequest

    print(f"\n  Loading {display_name}...")
    t0 = time.time()
    ranker = Ranker(model_name=model_name)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.2f}s")

    total_rerank_time = 0

    for query in QUERIES:
        candidates = get_bm25_candidates(query, N_CANDIDATES)
        if not candidates:
            print(f"\n  Q: \"{query}\" — no BM25 results")
            continue

        passages = [{"id": str(i), "text": c["text"][:500]} for i, c in enumerate(candidates)]

        t0 = time.time()
        reranked = ranker.rerank(RerankRequest(query=query, passages=passages))
        rerank_time = time.time() - t0
        total_rerank_time += rerank_time

        print(f"\n  Q: \"{query}\"  ({rerank_time*1000:.0f}ms)")
        for j, r in enumerate(reranked[:N_SHOW]):
            orig_idx = int(r["id"])
            c = candidates[orig_idx]
            src = f"{c.get('tutorial_display','')} ep{c.get('episode_num',0):02d}"
            ts = c.get("timestamp", "?")
            print(f"    [{j+1}] score={r['score']:.4f} | {src} @ {ts}")
            print(f"        {r['text'][:120]}...")

    avg_ms = total_rerank_time / len(QUERIES) * 1000
    print(f"\n  Avg rerank time: {avg_ms:.0f}ms per query ({N_CANDIDATES} candidates)")
    return avg_ms


def eval_baseline():
    """Show raw BM25 order (no reranking) as baseline."""
    print(f"\n  Baseline: BM25 only (no reranker)")

    for query in QUERIES:
        candidates = get_bm25_candidates(query, N_CANDIDATES)
        if not candidates:
            continue

        print(f"\n  Q: \"{query}\"")
        for j, c in enumerate(candidates[:N_SHOW]):
            src = f"{c.get('tutorial_display','')} ep{c.get('episode_num',0):02d}"
            ts = c.get("timestamp", "?")
            print(f"    [{j+1}] {src} @ {ts}")
            print(f"        {c['text'][:120]}...")


if __name__ == "__main__":
    print("=" * 70)
    print("  Reranker Eval on Real TutorialVault Data (211K chunks)")
    print("=" * 70)

    # Baseline
    eval_baseline()

    # Each reranker
    results = {}
    for display_name, model_name in RERANKER_MODELS:
        print(f"\n{'='*70}")
        print(f"  {display_name}")
        print(f"{'='*70}")
        try:
            avg_ms = eval_reranker(model_name, display_name)
            results[display_name] = avg_ms
        except Exception as e:
            print(f"  FAILED: {e}")
            results[display_name] = None

    # Summary
    print(f"\n{'='*70}")
    print(f"  SPEED SUMMARY")
    print(f"{'='*70}")
    for name, ms in results.items():
        if ms is not None:
            print(f"  {name:<30} {ms:>6.0f}ms / query")
        else:
            print(f"  {name:<30} FAILED")
