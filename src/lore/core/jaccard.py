"""Precomputed Jaccard similarity index on keyword + concept_tag sets.

For each chunk, stores the top-N most similar chunks by Jaccard coefficient
over their namespaced keyword/tag sets. Persisted to ~/.lore/jaccard_index.json.
Rebuilt on demand (same lazy singleton pattern as EntityGraph).
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import get_config
from .graph import _extract_terms


class JaccardIndex:
    """Chunk-to-chunk similarity index based on keyword/tag overlap."""

    def __init__(self, top_n: int = 10, min_jaccard: float = 0.1):
        self.top_n = top_n
        self.min_jaccard = min_jaccard
        self._index: dict[str, list[dict]] = {}
        self._cfg = get_config()
        self._index_path = self._cfg.data_dir / "jaccard_index.json"

    def build(self) -> "JaccardIndex":
        from .store import get_store

        store = get_store()

        chunk_terms: dict[str, tuple[set[str], str]] = {}

        for coll in store.list_collections():
            coll_name = coll["collection"]
            for c in store.iter_chunks(coll_name):
                chunk_id = c.get("id", "")
                if not chunk_id:
                    continue
                terms = _extract_terms(c)
                if terms:
                    chunk_terms[chunk_id] = (terms, coll_name)

        chunk_ids = list(chunk_terms.keys())
        n = len(chunk_ids)
        print(f"  [jaccard] Computing pairwise similarity for {n} chunks...")

        self._index = {}
        for i in range(n):
            id_a = chunk_ids[i]
            terms_a, _ = chunk_terms[id_a]
            neighbors = []
            for j in range(n):
                if i == j:
                    continue
                id_b = chunk_ids[j]
                terms_b, coll_b = chunk_terms[id_b]

                intersection = terms_a & terms_b
                if not intersection:
                    continue
                union_size = len(terms_a | terms_b)
                jaccard = len(intersection) / union_size

                if jaccard >= self.min_jaccard:
                    neighbors.append({
                        "chunk_id": id_b,
                        "jaccard": round(jaccard, 4),
                        "shared_terms": sorted(intersection),
                        "collection": coll_b,
                    })

            if neighbors:
                neighbors.sort(key=lambda x: -x["jaccard"])
                self._index[id_a] = neighbors[:self.top_n]

        chunks_with_neighbors = len(self._index)
        total_pairs = sum(len(v) for v in self._index.values())
        print(f"  [jaccard] {chunks_with_neighbors} chunks have neighbors, {total_pairs} total pairs")

        try:
            self.save()
        except OSError as e:
            print(f"  [jaccard] Warning: could not persist: {e}")

        return self

    def get_neighbors(self, chunk_id: str) -> list[dict]:
        """Return precomputed top-N Jaccard neighbors for a chunk."""
        return self._index.get(chunk_id, [])

    def stats(self) -> dict:
        chunks_with = len(self._index)
        total_pairs = sum(len(v) for v in self._index.values())
        avg_neighbors = total_pairs / chunks_with if chunks_with else 0
        return {
            "chunks_with_neighbors": chunks_with,
            "total_pairs": total_pairs,
            "avg_neighbors": round(avg_neighbors, 1),
            "top_n": self.top_n,
            "min_jaccard": self.min_jaccard,
        }

    def save(self):
        serializable = {}
        for chunk_id, neighbors in self._index.items():
            serializable[chunk_id] = neighbors

        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(json.dumps(serializable, indent=2))
        print(f"  [jaccard] Saved to {self._index_path}")

    def load(self) -> bool:
        if not self._index_path.exists():
            return False
        try:
            data = json.loads(self._index_path.read_text())
            self._index = data
            chunks_with = len(self._index)
            total_pairs = sum(len(v) for v in self._index.values())
            print(f"  [jaccard] Loaded {chunks_with} chunks, {total_pairs} pairs")
            return True
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [jaccard] Failed to load: {e}")
            return False


_jaccard_index: JaccardIndex | None = None


def get_jaccard_index(rebuild: bool = False) -> JaccardIndex:
    global _jaccard_index
    if _jaccard_index is None or rebuild:
        _jaccard_index = JaccardIndex()
        if not rebuild and _jaccard_index.load():
            return _jaccard_index
        _jaccard_index.build()
    return _jaccard_index


def invalidate_jaccard_index():
    global _jaccard_index
    _jaccard_index = None
