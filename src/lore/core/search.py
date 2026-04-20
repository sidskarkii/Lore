"""Search — hybrid retrieval with RRF fusion, FlashRank reranking, and parent expansion."""

from __future__ import annotations

import re
import time

from flashrank import Ranker, RerankRequest

from .chunk import fmt_timestamp
from .config import get_config
from .embed import embed_texts
from .store import Store, get_store

_ranker = None
_engine_instance: "SearchEngine | None" = None


def get_search_engine() -> "SearchEngine":
    """Return the process-wide SearchEngine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = SearchEngine()
    return _engine_instance


def _get_ranker():
    global _ranker
    if _ranker is not None:
        return _ranker

    cfg = get_config()
    model = cfg.get("search.reranker_model")
    if not model:
        return None

    print(f"  Loading reranker {model}...")
    _ranker = Ranker(model_name=model)
    return _ranker


def _rerank_with_scores(query: str, candidates: list[dict], n: int) -> list[tuple[dict, float]]:
    ranker = _get_ranker()
    if ranker is None:
        return [(c, 1.0) for c in candidates[:n]]

    passages = [{"id": str(i), "text": c["text"]} for i, c in enumerate(candidates)]
    reranked = ranker.rerank(RerankRequest(query=query, passages=passages))
    return [(candidates[int(r["id"])], r["score"]) for r in reranked[:n]]


def _rerank(query: str, candidates: list[dict], n: int) -> list[dict]:
    return [c for c, _ in _rerank_with_scores(query, candidates, n)]


def _rrf(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return scores


def _build_where(topic: str | None, subtopic: str | None) -> str | None:
    parts = []
    if topic:
        parts.append(f"topic = '{topic.lower()}'")
    if subtopic:
        parts.append(f"subtopic = '{subtopic.lower()}'")
    return " AND ".join(parts) if parts else None


_DECOMPOSE_PROMPT = (
    "Break this question into {max_queries} or fewer simpler search queries that "
    "together would find all the information needed to answer it. Return one query "
    "per line, nothing else. If the question is already simple, return it unchanged "
    "on one line.\n\nQuestion: {query}"
)

_STRIP_NUMBERING = re.compile(r"^\s*[\d]+[.):\-]\s*")


def _parse_sub_queries(text: str, max_queries: int) -> list[str]:
    queries = []
    for line in text.strip().splitlines():
        line = _STRIP_NUMBERING.sub("", line).strip().strip('"').strip("'")
        if line and len(line) > 5:
            queries.append(line)
    return queries[:max_queries]


class SearchEngine:
    def __init__(self, store: Store | None = None):
        self.store = store or get_store()
        self._cfg = get_config()

    def search(
        self,
        query: str,
        n_results: int = 5,
        topic: str | None = None,
        subtopic: str | None = None,
        expand: bool = True,
    ) -> list[dict]:
        cfg = self._cfg
        candidate_count = cfg.get("search.candidate_count", 30)
        rrf_k = cfg.get("search.rrf_k", 60)
        where = _build_where(topic, subtopic)

        t0 = time.perf_counter()

        # 1. Embed query
        query_vec = embed_texts([query])[0]
        t1 = time.perf_counter()
        print(f"  [search] embed:        {(t1-t0)*1000:.0f}ms")

        # 2. Vector search
        vec_results = self.store.vector_search(query_vec, n=candidate_count, where=where)
        t2 = time.perf_counter()
        print(f"  [search] vector:       {(t2-t1)*1000:.0f}ms  ({len(vec_results)} hits)")

        # 3. FTS search
        fts_results = self.store.fts_search(query, n=candidate_count, where=where)
        t3 = time.perf_counter()
        print(f"  [search] fts:          {(t3-t2)*1000:.0f}ms  ({len(fts_results)} hits)")

        # 4. RRF fusion
        vec_ids = [r["id"] for r in vec_results] if vec_results else []
        fts_ids = [r["id"] for r in fts_results] if fts_results else []
        rrf_scores = _rrf([vec_ids, fts_ids], k=rrf_k)

        all_by_id: dict[str, dict] = {}
        for r in vec_results + fts_results:
            if r["id"] not in all_by_id:
                all_by_id[r["id"]] = r

        candidates = sorted(
            [r for r in all_by_id.values() if r["id"] in rrf_scores],
            key=lambda r: rrf_scores.get(r["id"], 0),
            reverse=True,
        )[:n_results * 4]
        t4 = time.perf_counter()
        print(f"  [search] rrf:          {(t4-t3)*1000:.0f}ms  ({len(candidates)} candidates)")

        # 5. Rerank
        results = _rerank(query, candidates, n_results)
        t5 = time.perf_counter()
        print(f"  [search] rerank:       {(t5-t4)*1000:.0f}ms  ({len(results)} results)")

        # 6. Parent window expansion
        if expand:
            results = [self._expand_to_parent(r) for r in results]
        t6 = time.perf_counter()
        print(f"  [search] expand:       {(t6-t5)*1000:.0f}ms")
        print(f"  [search] TOTAL:        {(t6-t0)*1000:.0f}ms")

        return results

    def _expand_to_parent(self, chunk: dict) -> dict:
        has_timestamps = int(chunk.get("start_sec", 0)) > 0 or int(chunk.get("end_sec", 0)) > 0
        has_chunk_index = "chunk_index" in chunk

        if has_timestamps:
            return self._expand_temporal(chunk)
        elif has_chunk_index:
            return self._expand_by_index(chunk)
        return chunk

    def _expand_temporal(self, chunk: dict) -> dict:
        """Expand using time window (video/audio sources)."""
        window = self._cfg.get("search.parent_window_sec", 150)
        try:
            center = int(chunk["start_sec"])
            neighbors = self.store.get_neighbors(
                collection=chunk["collection"],
                episode_num=chunk["episode_num"],
                start_sec=max(0, center - window),
                end_sec=center + window,
            )
            if not neighbors:
                return chunk

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

    def _expand_by_index(self, chunk: dict) -> dict:
        """Expand using chunk index (document/code sources)."""
        try:
            idx = int(chunk.get("chunk_index", 0))
            neighbors = self.store.get_neighbors_by_index(
                collection=chunk["collection"],
                episode_num=chunk["episode_num"],
                chunk_index_start=max(0, idx - 2),
                chunk_index_end=idx + 2,
            )
            if not neighbors or len(neighbors) <= 1:
                return chunk

            parts = [row["text"] for row in neighbors]
            expanded = dict(chunk)
            expanded["text"] = "\n\n".join(parts)
            return expanded
        except Exception:
            return chunk

    def search_multi_hop(self, query, provider, n_results=5, topic=None, subtopic=None):
        cfg = self._cfg
        max_queries = cfg.get("search.multi_hop_max_queries", 4)
        threshold = cfg.get("search.multi_hop_relevance_threshold", 0.1)

        if provider is None:
            return self.search(query, n_results, topic, subtopic)

        try:
            prompt = _DECOMPOSE_PROMPT.format(max_queries=max_queries, query=query)
            raw = provider.chat([{"role": "user", "content": prompt}])
            sub_queries = _parse_sub_queries(raw, max_queries)
        except Exception:
            return self.search(query, n_results, topic, subtopic)

        if len(sub_queries) <= 1:
            return self.search(query, n_results, topic, subtopic)

        print(f"  [multi-hop] {len(sub_queries)} sub-queries:")
        for i, sq in enumerate(sub_queries, 1):
            print(f"    {i}. {sq}")

        per_query_n = max(n_results, 5)
        all_ranked_ids: list[list[str]] = []
        all_by_id: dict[str, dict] = {}

        for sq in sub_queries:
            results = self.search(sq, n_results=per_query_n, topic=topic, subtopic=subtopic)
            ranked_ids = [r["id"] for r in results]
            for r in results:
                if r["id"] not in all_by_id:
                    all_by_id[r["id"]] = r
            all_ranked_ids.append(ranked_ids)

        if not all_by_id:
            return self.search(query, n_results, topic, subtopic)

        rrf_k = cfg.get("search.rrf_k", 60)
        rrf_scores = _rrf(all_ranked_ids, k=rrf_k)
        candidates = sorted(all_by_id.values(), key=lambda r: rrf_scores.get(r["id"], 0), reverse=True)[:n_results * 4]

        scored = _rerank_with_scores(query, candidates, n_results * 2)
        results = [c for c, score in scored if score >= threshold] or [c for c, _ in scored[:n_results]]
        results = results[:n_results]
        results = [self._expand_to_parent(r) for r in results]

        print(f"  [multi-hop] {len(results)} final results from {len(all_by_id)} candidates")
        return results
