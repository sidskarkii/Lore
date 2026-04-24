"""Search — hybrid retrieval with RRF fusion, FlashRank reranking, and parent expansion."""

from __future__ import annotations

import re
import time

from flashrank import Ranker, RerankRequest

import json

import math

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

    cache_dir = str(cfg.data_dir / "models")
    print(f"  Loading reranker {model}...")
    _ranker = Ranker(model_name=model, cache_dir=cache_dir)
    return _ranker


def _rerank_with_scores(query: str, candidates: list[dict], n: int) -> list[tuple[dict, float]]:
    ranker = _get_ranker()
    if ranker is None:
        return [(c, 1.0) for c in candidates[:n]]

    try:
        passages = [{"id": str(i), "text": c["text"]} for i, c in enumerate(candidates)]
        reranked = ranker.rerank(RerankRequest(query=query, passages=passages))
        return [(candidates[int(r["id"])], r["score"]) for r in reranked[:n]]
    except Exception as e:
        print(f"  [search] Reranker failed, using RRF order: {e}")
        return [(c, 1.0) for c in candidates[:n]]


def _rerank(query: str, candidates: list[dict], n: int) -> list[dict]:
    return [c for c, _ in _rerank_with_scores(query, candidates, n)]


def _rrf(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return scores


def _esc(val: str) -> str:
    return val.replace("'", "''")


def _build_where(topic: str | None, subtopic: str | None) -> str | None:
    parts = []
    if topic:
        parts.append(f"topic = '{_esc(topic.lower())}'")
    if subtopic:
        parts.append(f"subtopic = '{_esc(subtopic.lower())}'")
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


def _extract_query_entities(query: str) -> set[str]:
    """Extract entity names from the search query using spaCy, expanded via entity index."""
    try:
        from .enrich import _get_nlp
        nlp = _get_nlp()
        doc = nlp(query)
        entities = set()
        for ent in doc.ents:
            entities.add(ent.text.lower())
        for token in doc:
            if token.pos_ == "PROPN":
                entities.add(token.text.lower())

        try:
            from .entities import get_entity_index
            idx = get_entity_index()
            expanded = set()
            for ent in entities:
                cluster = idx.resolve(ent)
                if cluster:
                    expanded.update(v.lower() for v in cluster.variants)
                else:
                    expanded.add(ent)
            return expanded
        except Exception:
            return entities
    except Exception:
        return set()


def _entity_rank(candidates: list[dict], query_entities: set[str]) -> list[str]:
    """Rank candidates by entity overlap with query entities (entity-index-aware)."""
    try:
        from .entities import get_entity_index
        idx = get_entity_index()
    except Exception:
        idx = None

    scored = []
    for c in candidates:
        try:
            ents_raw = c.get("entities", "")
            if ents_raw:
                ents = json.loads(ents_raw) if isinstance(ents_raw, str) else ents_raw
                chunk_entities = set()
                for e in ents:
                    if not isinstance(e, dict):
                        continue
                    name = e.get("name", "").lower()
                    chunk_entities.add(name)
                    if idx:
                        cluster = idx.resolve(name)
                        if cluster:
                            chunk_entities.update(v.lower() for v in cluster.variants)
            else:
                chunk_entities = set()
        except (json.JSONDecodeError, TypeError):
            chunk_entities = set()

        kw_raw = c.get("keywords", "")
        chunk_keywords = {k.strip().lower() for k in kw_raw.split(",")} if kw_raw else set()

        ct_raw = c.get("concept_tags", "")
        chunk_concepts = {k.strip().lower() for k in ct_raw.split(",")} if ct_raw else set()

        overlap = len(query_entities & (chunk_entities | chunk_keywords | chunk_concepts))
        scored.append((c["id"], overlap))

    scored.sort(key=lambda x: -x[1])
    return [cid for cid, score in scored if score > 0]


def _wilson_score(fetches: int, ignores: int, z: float = 1.96) -> float:
    """Wilson Score lower bound. Returns 0.5 (neutral) when no data."""
    n = fetches + ignores
    if n == 0:
        return 0.5
    p = fetches / n
    return (p + z * z / (2 * n) - z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / (1 + z * z / n)


def _apply_rating_boost(results: list[dict], weight: float = 0.1) -> list[dict]:
    """Adjust reranker scores using Wilson Score from chunk_ratings + importance."""
    try:
        from .database import get_database
        db = get_database()
        chunk_ids = [r.get("id", "") for r in results]
        ratings = db.get_chunk_ratings_batch(chunk_ids)
    except Exception:
        ratings = {}

    for r in results:
        chunk_id = r.get("id", "")
        rating = ratings.get(chunk_id)
        if rating:
            ws = _wilson_score(rating["fetches"] + rating["explicit_up"] * 3,
                               rating["ignores"] + rating["explicit_down"] * 3)
            r["_score"] = r.get("_score", 0) + weight * (ws - 0.5)

        importance = r.get("importance", 3)
        if importance and importance != 3:
            r["_score"] = r.get("_score", 0) + weight * 0.5 * (max(1, min(5, importance)) - 3) / 2

    results.sort(key=lambda r: r.get("_score", 0), reverse=True)
    return results


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
        session_id: str | None = None,
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

        # 4. RRF fusion (vector + FTS + entity overlap)
        all_by_id: dict[str, dict] = {}
        for r in vec_results + fts_results:
            if r["id"] not in all_by_id:
                all_by_id[r["id"]] = r

        vec_ids = [r["id"] for r in vec_results] if vec_results else []
        fts_ids = [r["id"] for r in fts_results] if fts_results else []

        query_entities = _extract_query_entities(query)
        if query_entities:
            entity_ids = _entity_rank(list(all_by_id.values()), query_entities)
            rrf_scores = _rrf([vec_ids, fts_ids, entity_ids], k=rrf_k)
            print(f"  [search] entities:     {len(query_entities)} query entities, {len(entity_ids)} matches")
        else:
            rrf_scores = _rrf([vec_ids, fts_ids], k=rrf_k)

        candidates = sorted(
            [r for r in all_by_id.values() if r["id"] in rrf_scores],
            key=lambda r: rrf_scores.get(r["id"], 0),
            reverse=True,
        )[:n_results * 4]
        t4 = time.perf_counter()
        print(f"  [search] rrf:          {(t4-t3)*1000:.0f}ms  ({len(candidates)} candidates)")

        # 5. Rerank
        scored = _rerank_with_scores(query, candidates, n_results)
        results = []
        for c, score in scored:
            r = dict(c)
            r["_score"] = round(score, 4)
            results.append(r)
        t5 = time.perf_counter()
        print(f"  [search] rerank:       {(t5-t4)*1000:.0f}ms  ({len(results)} results)")

        # 6. Rating + importance boost
        results = _apply_rating_boost(results)

        # 7. Session-aware deprioritization (with TTL)
        if session_id:
            try:
                from .database import get_database
                ttl = cfg.get("search.session_ttl_minutes", 30)
                fetched_ids = get_database().get_session_fetched_ids(session_id, ttl_minutes=ttl)
            except Exception:
                fetched_ids = set()

            if fetched_ids:
                for r in results:
                    if r.get("id") in fetched_ids:
                        r["_score"] = r.get("_score", 0) * 0.5
                results.sort(key=lambda r: r.get("_score", 0), reverse=True)
                print(f"  [search] session:      {len(fetched_ids)} fetched, deprioritized")

        # 8. Parent window expansion
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
        filtered = [(c, s) for c, s in scored if s >= threshold] or scored[:n_results]
        results = []
        for c, score in filtered[:n_results]:
            r = dict(c)
            r["_score"] = round(score, 4)
            results.append(r)
        results = [self._expand_to_parent(r) for r in results]

        print(f"  [multi-hop] {len(results)} final results from {len(all_by_id)} candidates")
        return results
