"""CrossSourceIndex — postings-based candidate generation for find_related.

Pre-parses and indexes chunk features (entities, keywords, tags) at build
time so find_related can do O(candidates) scoring instead of O(all_chunks).
Lazy singleton, rebuilt after ingest/delete via invalidation hooks.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

from .config import get_config


@dataclass(frozen=True, slots=True)
class ChunkFeatures:
    chunk_id: str
    collection: str
    episode_title: str
    entities: frozenset[str]
    keywords: frozenset[str]
    tags: frozenset[str]

    @property
    def terms(self) -> frozenset[str]:
        return frozenset({*(f"kw:{k}" for k in self.keywords), *(f"tag:{t}" for t in self.tags)})


def dice(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    """Symmetric Dice coefficient: 2*|A∩B| / (|A|+|B|)."""
    if not a or not b:
        return 0.0
    return (2.0 * len(set(a) & set(b))) / (len(a) + len(b))


def _parse_csv_or_json(raw) -> frozenset[str]:
    """Parse a field that may be comma-separated string, JSON array, or list."""
    if not raw:
        return frozenset()
    if isinstance(raw, list):
        return frozenset(v.strip().lower() for v in raw if isinstance(v, str) and v.strip())
    if isinstance(raw, str):
        try:
            values = json.loads(raw)
            if isinstance(values, list):
                return frozenset(v.strip().lower() for v in values if isinstance(v, str) and v.strip())
        except (json.JSONDecodeError, ValueError):
            pass
        return frozenset(v.strip().lower() for v in raw.split(",") if v.strip())
    return frozenset()


def _parse_entities(chunk: dict, entity_index) -> frozenset[str]:
    """Resolve chunk entities to canonical names via the entity index."""
    ents_raw = chunk.get("entities", "")
    if not ents_raw:
        return frozenset()
    try:
        ents = json.loads(ents_raw) if isinstance(ents_raw, str) else ents_raw
        result = set()
        for e in ents:
            if isinstance(e, dict) and e.get("name"):
                cl = entity_index.resolve(e["name"])
                if cl:
                    result.add(cl.canonical)
        return frozenset(result)
    except (json.JSONDecodeError, TypeError):
        return frozenset()


class CrossSourceIndex:
    """In-memory index of chunk features with inverted postings."""

    def __init__(self):
        self.by_chunk: dict[str, ChunkFeatures] = {}
        self.entity_postings: dict[str, set[str]] = defaultdict(set)
        self.term_postings: dict[str, set[str]] = defaultdict(set)

    def build(self) -> "CrossSourceIndex":
        from .store import get_store
        from .entities import get_entity_index

        store = get_store()
        idx = get_entity_index()

        self.by_chunk.clear()
        self.entity_postings = defaultdict(set)
        self.term_postings = defaultdict(set)

        for coll in store.list_collections():
            for chunk in store.iter_chunks(coll["collection"]):
                chunk_id = chunk.get("id", "")
                if not chunk_id:
                    continue

                entities = _parse_entities(chunk, idx)
                keywords = _parse_csv_or_json(chunk.get("keywords", ""))
                tags = _parse_csv_or_json(chunk.get("concept_tags", ""))

                feats = ChunkFeatures(
                    chunk_id=chunk_id,
                    collection=chunk.get("collection", ""),
                    episode_title=chunk.get("episode_title", ""),
                    entities=entities,
                    keywords=keywords,
                    tags=tags,
                )

                self.by_chunk[chunk_id] = feats
                for e in entities:
                    self.entity_postings[e].add(chunk_id)
                for t in feats.terms:
                    self.term_postings[t].add(chunk_id)

        print(f"  [cross_index] {len(self.by_chunk)} chunks, "
              f"{len(self.entity_postings)} entity postings, "
              f"{len(self.term_postings)} term postings")
        return self

    def find_related(self, chunk_id: str, collection: str | None = None,
                     n_results: int = 10, max_term_df: int = 500) -> dict:
        """Fused scoring: 0.60*entity_dice + 0.25*kw_dice + 0.15*tag_dice."""
        source = self.by_chunk.get(chunk_id)
        if not source:
            return {"success": False, "error": f"Chunk not in index: {chunk_id}"}

        if not source.entities and not source.keywords and not source.tags:
            return {"success": True, "chunk_id": chunk_id,
                    "message": "No entities or keywords found on this chunk", "results": []}

        candidate_ids: set[str] = set()
        for e in source.entities:
            candidate_ids.update(self.entity_postings.get(e, set()))
        for t in source.terms:
            posting = self.term_postings.get(t, set())
            if len(posting) <= max_term_df:
                candidate_ids.update(posting)
        candidate_ids.discard(chunk_id)

        if collection:
            candidate_ids = {cid for cid in candidate_ids
                             if self.by_chunk.get(cid) and self.by_chunk[cid].collection == collection}

        scored = []
        for cid in candidate_ids:
            cand = self.by_chunk.get(cid)
            if not cand:
                continue

            entity_d = dice(source.entities, cand.entities)
            kw_d = dice(source.keywords, cand.keywords)
            tag_d = dice(source.tags, cand.tags)

            combined = 0.60 * entity_d + 0.25 * kw_d + 0.15 * tag_d

            if combined > 0:
                scored.append({
                    "chunk_id": cid,
                    "collection": cand.collection,
                    "episode_title": cand.episode_title,
                    "score": round(combined, 4),
                    "shared_entities": sorted(source.entities & cand.entities),
                    "shared_keywords": sorted(source.keywords & cand.keywords),
                    "shared_tags": sorted(source.tags & cand.tags),
                })

        scored.sort(key=lambda x: -x["score"])
        total_found = len(scored)

        return {
            "success": True,
            "chunk_id": chunk_id,
            "query_entities": sorted(source.entities),
            "query_keywords": len(source.keywords),
            "query_tags": len(source.tags),
            "total_related": total_found,
            "returned": min(n_results, total_found),
            "results": scored[:n_results],
        }

    def find_by_entity(self, entity: str, collection: str | None = None,
                       n_results: int = 10) -> dict:
        """Entity-only lookup via postings — O(matches), no full scan."""
        from .entities import get_entity_index

        idx = get_entity_index()
        cluster = idx.resolve(entity)
        canonical = cluster.canonical if cluster else entity.lower()

        chunk_ids = self.entity_postings.get(canonical, set())
        if not chunk_ids and not cluster:
            chunk_ids = self.entity_postings.get(entity.lower(), set())

        results = []
        for cid in chunk_ids:
            feats = self.by_chunk.get(cid)
            if not feats:
                continue
            if collection and feats.collection != collection:
                continue
            results.append({
                "chunk_id": cid,
                "collection": feats.collection,
                "episode_title": feats.episode_title,
                "shared_entities": [canonical],
                "match_count": 1,
            })

        results.sort(key=lambda x: x["chunk_id"])
        total_found = len(results)
        return {
            "success": True,
            "query_entities": [canonical],
            "total_related": total_found,
            "returned": min(n_results, total_found),
            "results": results[:n_results],
        }


_cross_index: CrossSourceIndex | None = None


def get_cross_index(rebuild: bool = False) -> CrossSourceIndex:
    global _cross_index
    if _cross_index is None or rebuild:
        _cross_index = CrossSourceIndex()
        _cross_index.build()
    return _cross_index


def invalidate_cross_index():
    global _cross_index
    _cross_index = None
