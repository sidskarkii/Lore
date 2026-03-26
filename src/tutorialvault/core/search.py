"""Search — hybrid retrieval with RRF fusion, FlashRank reranking, and parent expansion."""

from __future__ import annotations

from .chunk import fmt_timestamp
from .config import get_config
from .embed import embed_texts
from .store import Store

_ranker = None


def _get_ranker():
    """Lazy-load FlashRank reranker."""
    global _ranker
    if _ranker is not None:
        return _ranker

    cfg = get_config()
    model = cfg.get("search.reranker_model")
    if not model:
        return None

    from flashrank import Ranker
    print(f"  Loading reranker {model}...")
    _ranker = Ranker(model_name=model)
    return _ranker


def _rerank(query: str, candidates: list[dict], n: int) -> list[dict]:
    """Rerank candidates using FlashRank. Returns top n."""
    ranker = _get_ranker()
    if ranker is None:
        return candidates[:n]

    from flashrank import RerankRequest

    passages = [
        {"id": str(i), "text": c["text"][:500]}
        for i, c in enumerate(candidates)
    ]
    reranked = ranker.rerank(RerankRequest(query=query, passages=passages))
    return [candidates[int(r["id"])] for r in reranked[:n]]


def _rrf(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion across multiple ranked ID lists."""
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return scores


def _build_where(topic: str | None, subtopic: str | None) -> str | None:
    """Build a LanceDB WHERE clause from filters."""
    parts = []
    if topic:
        parts.append(f"topic = '{topic.lower()}'")
    if subtopic:
        parts.append(f"subtopic = '{subtopic.lower()}'")
    return " AND ".join(parts) if parts else None


class SearchEngine:
    """Hybrid search with RRF fusion, FlashRank reranking, parent expansion."""

    def __init__(self, store: Store | None = None):
        self.store = store or Store()
        self._cfg = get_config()

    def search(
        self,
        query: str,
        n_results: int = 5,
        topic: str | None = None,
        subtopic: str | None = None,
    ) -> list[dict]:
        """Run hybrid search and return ranked, expanded results.

        Pipeline:
        1. Embed query
        2. Vector search (semantic)
        3. FTS search (BM25 keyword)
        4. RRF fusion
        5. FlashRank cross-encoder reranking
        6. Parent window expansion
        """
        cfg = self._cfg
        candidate_count = cfg.get("search.candidate_count", 30)
        rrf_k = cfg.get("search.rrf_k", 60)
        where = _build_where(topic, subtopic)

        # 1. Embed query
        query_vec = embed_texts([query])[0]

        # 2. Vector search
        vec_results = self.store.vector_search(query_vec, n=candidate_count, where=where)

        # 3. FTS search
        fts_results = self.store.fts_search(query, n=candidate_count, where=where)

        # 4. RRF fusion
        vec_ids = [r["id"] for r in vec_results] if vec_results else []
        fts_ids = [r["id"] for r in fts_results] if fts_results else []
        rrf_scores = _rrf([vec_ids, fts_ids], k=rrf_k)

        # Merge and deduplicate
        all_by_id: dict[str, dict] = {}
        for r in vec_results + fts_results:
            if r["id"] not in all_by_id:
                all_by_id[r["id"]] = r

        # Sort by RRF score
        candidates = sorted(
            [r for r in all_by_id.values() if r["id"] in rrf_scores],
            key=lambda r: rrf_scores.get(r["id"], 0),
            reverse=True,
        )[:n_results * 4]

        # 5. FlashRank reranking
        results = _rerank(query, candidates, n_results)

        # 6. Parent window expansion
        results = [self._expand_to_parent(r) for r in results]

        return results

    def _expand_to_parent(self, chunk: dict) -> dict:
        """Expand a matched chunk to its surrounding parent window."""
        window = self._cfg.get("search.parent_window_sec", 150)
        try:
            center = int(chunk["start_sec"])
            win_start = max(0, center - window)
            win_end = center + window

            neighbors = self.store.get_neighbors(
                collection=chunk["collection"],
                episode_num=chunk["episode_num"],
                start_sec=win_start,
                end_sec=win_end,
            )

            if not neighbors:
                return chunk

            # Deduplicate overlapping text, preserve order
            seen: set[tuple[int, int]] = set()
            parts: list[str] = []
            for row in neighbors:
                key = (row["start_sec"], row["end_sec"])
                if key not in seen:
                    seen.add(key)
                    parts.append(row["text"])

            expanded = dict(chunk)
            expanded["text"] = "\n\n".join(parts)
            expanded["start_sec"] = int(neighbors[0]["start_sec"])
            expanded["end_sec"] = int(neighbors[-1]["end_sec"])
            expanded["timestamp"] = fmt_timestamp(neighbors[0]["start_sec"])
            return expanded
        except Exception:
            return chunk
