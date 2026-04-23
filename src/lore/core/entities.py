"""Entity index — fuzzy merging, normalization, and canonical entity resolution.

Collects raw NER entities from all chunks, normalizes surface forms,
filters noise, and clusters variants into canonical entities using
Jaro-Winkler similarity (rapidfuzz). The index persists to
~/.lore/entity_index.json and is rebuilt on demand.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import JaroWinkler

from .config import get_config

_STOP_ENTITIES = {
    "king", "majesty", "state", "palace", "army", "emperor",
    "lord", "prince", "minister", "general", "master", "sir",
    "strong", "ford", "moon", "earth", "sun", "god",
}

_STRUCTURAL_RE = re.compile(
    r"^(?:chapter|section|figure|fig\.?|table|appendix|part|volume|page|exhibit)"
    r"\s+(?:\d[\d.a-z]*|[a-z]|[ivxlc]+)$",
    re.IGNORECASE,
)

_ROMAN_ONLY_RE = re.compile(
    r"^(?:I{1,4}V?|VI{0,4}|IX|XI{0,3}|XI?V|XV?I{0,3})$"
)

_POSSESSIVE_RE = re.compile(r"['']s$")
_LEADING_NOISE = re.compile(r"^(?:the|a|an|of|in|on|at|to|for|by|with|from|thereupon|introduction)\s+", re.IGNORECASE)

_KNOWN_GPES = {
    "france", "london", "paris", "china", "rome", "spain", "england",
    "germany", "italy", "russia", "japan", "india", "egypt", "greece",
    "turkey", "persia", "austria", "prussia", "holland", "brazil",
    "mexico", "canada", "australia", "ireland", "scotland",
    "venice", "milan", "naples", "amsterdam",
    "berlin", "vienna", "moscow", "beijing", "tokyo",
    "new york", "los angeles", "chicago", "boston", "philadelphia",
    "athens", "carthage", "constantinople", "babylon", "jerusalem",
    "africa", "europe", "asia", "america", "arabia",
}

_COARSE_TYPE = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "PLACE", "LOC": "PLACE", "FAC": "PLACE",
    "WORK_OF_ART": "WORK", "LAW": "WORK",
    "EVENT": "EVENT",
    "PRODUCT": "PRODUCT", "TECH": "PRODUCT",
    "CONCEPT": "CONCEPT",
    "METRIC": "METRIC",
    "NORP": "GROUP",
}


def _correct_type(name: str, etype: str) -> str:
    """Conservative type correction via gazetteer."""
    if name.lower() in _KNOWN_GPES and etype in ("PERSON", "ORG"):
        return "GPE"
    return etype


def _normalize(name: str) -> str:
    """Normalize an entity name for comparison."""
    name = unicodedata.normalize("NFKD", name)
    name = name.replace("\n", " ").replace("\r", " ")
    name = re.sub(r"\s+", " ", name).strip()
    name = _POSSESSIVE_RE.sub("", name)
    name = _LEADING_NOISE.sub("", name)
    name = name.strip().strip("'\".,;:!?()[]{}").strip()
    return name


def _should_filter(name: str, normalized: str) -> bool:
    """Return True if this entity should be discarded as noise."""
    if len(normalized) < 2:
        return True
    if len(normalized) <= 3 and normalized.isupper():
        return True
    if _STRUCTURAL_RE.match(normalized):
        return True
    if _ROMAN_ONLY_RE.match(normalized):
        return True
    if normalized.lower() in _STOP_ENTITIES:
        return True
    if all(c in ".-()[]0123456789 " for c in normalized):
        return True
    if normalized.count(" ") > 6:
        return True
    return False


def _merge_threshold(name: str, base: float = 85.0) -> float:
    """Adaptive merge threshold based on name length and token count."""
    token_count = len(name.split())
    if token_count == 1:
        if len(name) <= 5:
            return 100.0
        elif len(name) <= 8:
            return 95.0
        else:
            return 93.0
    if token_count >= 2:
        return 90.0
    return base


def _coarse_type_compatible(type_a: str, type_b: str) -> bool:
    """Check if two entity types are compatible for merging."""
    ca = _COARSE_TYPE.get(type_a, type_a)
    cb = _COARSE_TYPE.get(type_b, type_b)
    if ca == cb:
        return True
    if "UNKNOWN" in (ca, cb):
        return True
    return False


def _jaro_winkler(a: str, b: str) -> float:
    """Jaro-Winkler similarity (0.0-1.0 scale), scaled to 0-100."""
    return JaroWinkler.similarity(a.lower(), b.lower()) * 100


class EntityCluster:
    """A group of entity name variants that refer to the same real-world entity."""

    __slots__ = ("canonical", "entity_type", "variants", "sources", "count", "_type_counts")

    def __init__(self, canonical: str, entity_type: str):
        self.canonical = canonical
        self.entity_type = entity_type
        self.variants: set[str] = {canonical}
        self.sources: set[str] = set()
        self.count = 1
        self._type_counts: Counter[str] = Counter({entity_type: 1})

    def merge(self, name: str, entity_type: str, source: str = ""):
        self.variants.add(name)
        self.count += 1
        if source:
            self.sources.add(source)
        self._type_counts[entity_type] += 1
        self.entity_type = self._type_counts.most_common(1)[0][0]

    def to_dict(self) -> dict:
        return {
            "canonical": self.canonical,
            "type": self.entity_type,
            "variants": sorted(self.variants),
            "sources": sorted(self.sources),
            "count": self.count,
            "type_counts": dict(self._type_counts),
        }


class EntityIndex:
    """Fuzzy-merged entity index across all collections."""

    def __init__(self, threshold: float = 85.0, min_count: int = 1):
        self.threshold = threshold
        self.min_count = min_count
        self.clusters: list[EntityCluster] = []
        self._variant_map: dict[str, int] = {}
        self._cfg = get_config()
        self._index_path = self._cfg.data_dir / "entity_index.json"

    def _find_best_cluster(self, normalized: str, entity_type: str = "") -> tuple[int, float]:
        """Find the best matching cluster for a normalized entity name (type-aware)."""
        best_idx = -1
        best_score = 0.0
        incoming_threshold = _merge_threshold(normalized, self.threshold)
        for i, cluster in enumerate(self.clusters):
            if entity_type and not _coarse_type_compatible(entity_type, cluster.entity_type):
                continue
            cluster_threshold = _merge_threshold(cluster.canonical, self.threshold)
            required = max(incoming_threshold, cluster_threshold)
            score = _jaro_winkler(normalized, cluster.canonical.lower())
            if score >= required and score > best_score:
                best_score = score
                best_idx = i
            for variant in cluster.variants:
                variant_threshold = _merge_threshold(variant, self.threshold)
                vrequired = max(incoming_threshold, variant_threshold)
                vscore = _jaro_winkler(normalized, variant.lower())
                if vscore >= vrequired and vscore > best_score:
                    best_score = vscore
                    best_idx = i
        return best_idx, best_score

    def build_from_store(self) -> "EntityIndex":
        """Rebuild the entity index from all chunks in the store."""
        from .store import get_store
        store = get_store()

        raw_entities: list[tuple[str, str, str]] = []

        for coll in store.list_collections():
            collection_name = coll["collection"]
            try:
                chunks = store.get_all_chunks(collection_name)
            except Exception:
                continue

            for chunk in chunks:
                ents_raw = chunk.get("entities", "")
                if not ents_raw:
                    continue
                try:
                    ents = json.loads(ents_raw) if isinstance(ents_raw, str) else ents_raw
                except (json.JSONDecodeError, TypeError):
                    continue

                for e in ents:
                    if not isinstance(e, dict):
                        continue
                    name = e.get("name", "").strip()
                    etype = e.get("type", "UNKNOWN")
                    if name:
                        raw_entities.append((name, etype, collection_name))

        print(f"  [entities] Collected {len(raw_entities)} raw entity mentions")
        self._cluster_entities(raw_entities)
        try:
            self.save()
        except OSError as e:
            print(f"  [entities] Warning: could not persist index: {e}")
        return self

    def build_from_archive(self) -> "EntityIndex":
        """Rebuild the entity index from archived chunk data."""
        archive_dir = self._cfg.archive_dir
        if not archive_dir.exists():
            print("  [entities] No archive directory found")
            return self

        raw_entities: list[tuple[str, str, str]] = []

        for coll_dir in archive_dir.iterdir():
            if not coll_dir.is_dir():
                continue
            chunks_file = coll_dir / "chunks.json"
            if not chunks_file.exists():
                continue

            collection_name = coll_dir.name
            try:
                chunks = json.loads(chunks_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            for chunk in chunks:
                ents_raw = chunk.get("entities", "")
                if not ents_raw:
                    continue
                try:
                    ents = json.loads(ents_raw) if isinstance(ents_raw, str) else ents_raw
                except (json.JSONDecodeError, TypeError):
                    continue

                for e in ents:
                    if not isinstance(e, dict):
                        continue
                    name = e.get("name", "").strip()
                    etype = e.get("type", "UNKNOWN")
                    if name:
                        raw_entities.append((name, etype, collection_name))

        print(f"  [entities] Collected {len(raw_entities)} raw entity mentions from archive")
        self._cluster_entities(raw_entities)
        try:
            self.save()
        except OSError as e:
            print(f"  [entities] Warning: could not persist index: {e}")
        return self

    def _cluster_entities(self, raw_entities: list[tuple[str, str, str]]):
        """Normalize, filter, type-correct, and cluster raw entities."""
        name_counts: Counter[str] = Counter()
        name_info: dict[str, list[tuple[str, str, str]]] = {}

        for raw_name, etype, source in raw_entities:
            normalized = _normalize(raw_name)
            if _should_filter(raw_name, normalized):
                continue
            etype = _correct_type(normalized, etype)
            name_counts[normalized] += 1
            name_info.setdefault(normalized, []).append((raw_name, etype, source))

        sorted_names = name_counts.most_common()
        print(f"  [entities] {len(sorted_names)} unique names after normalization + filtering")

        self.clusters = []
        self._variant_map = {}

        for normalized, count in sorted_names:
            infos = name_info[normalized]
            type_counts = Counter(etype for _, etype, _ in infos)
            majority_type = type_counts.most_common(1)[0][0]
            sources = {src for _, _, src in infos}

            threshold = _merge_threshold(normalized, self.threshold)
            best_idx, best_score = self._find_best_cluster(normalized, majority_type)

            if best_score >= threshold and best_idx >= 0:
                cluster = self.clusters[best_idx]
                for raw_name, etype, source in infos:
                    cluster.merge(_normalize(raw_name), _correct_type(_normalize(raw_name), etype), source)
                self._variant_map[normalized] = best_idx
            else:
                idx = len(self.clusters)
                cluster = EntityCluster(normalized, majority_type)
                cluster.count = count
                cluster.sources = sources
                for raw_name, _, _ in infos:
                    cluster.variants.add(_normalize(raw_name))
                self.clusters.append(cluster)
                self._variant_map[normalized] = idx

        self.clusters = [c for c in self.clusters if c.count >= self.min_count]
        self._rebuild_variant_map()

        merged = sum(1 for c in self.clusters if len(c.variants) > 1)
        print(f"  [entities] {len(self.clusters)} clusters ({merged} merged from multiple variants)")

    def _rebuild_variant_map(self):
        self._variant_map = {}
        for i, cluster in enumerate(self.clusters):
            for variant in cluster.variants:
                self._variant_map[variant.lower()] = i

    def resolve(self, name: str, entity_type: str = "") -> EntityCluster | None:
        """Resolve an entity name to its canonical cluster."""
        normalized = _normalize(name).lower()
        idx = self._variant_map.get(normalized)
        if idx is not None and idx < len(self.clusters):
            return self.clusters[idx]

        threshold = _merge_threshold(normalized, self.threshold)
        best_idx, best_score = self._find_best_cluster(normalized, entity_type)
        if best_score >= threshold and best_idx >= 0:
            return self.clusters[best_idx]
        return None

    def get_cross_source_entities(self) -> list[EntityCluster]:
        """Return entities that appear in multiple collections."""
        return [c for c in self.clusters if len(c.sources) > 1]

    def save(self):
        """Persist the entity index to disk."""
        data = {
            "threshold": self.threshold,
            "total_clusters": len(self.clusters),
            "clusters": [c.to_dict() for c in self.clusters],
        }
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(json.dumps(data, indent=2))
        print(f"  [entities] Saved {len(self.clusters)} clusters to {self._index_path}")

    def load(self) -> bool:
        """Load the entity index from disk. Returns True if loaded."""
        if not self._index_path.exists():
            return False
        try:
            data = json.loads(self._index_path.read_text())
            self.threshold = data.get("threshold", self.threshold)
            self.clusters = []
            for cd in data.get("clusters", []):
                cluster = EntityCluster(cd["canonical"], cd["type"])
                cluster.variants = set(cd.get("variants", []))
                cluster.sources = set(cd.get("sources", []))
                cluster.count = cd.get("count", 1)
                if "type_counts" in cd:
                    cluster._type_counts = Counter(cd["type_counts"])
                self.clusters.append(cluster)
            self._rebuild_variant_map()
            print(f"  [entities] Loaded {len(self.clusters)} clusters from index")
            return True
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"  [entities] Failed to load index: {e}")
            return False

    def stats(self) -> dict:
        """Return summary statistics about the entity index."""
        cross_source = self.get_cross_source_entities()
        type_counts = Counter(c.entity_type for c in self.clusters)
        return {
            "total_clusters": len(self.clusters),
            "total_variants": sum(len(c.variants) for c in self.clusters),
            "merged_clusters": sum(1 for c in self.clusters if len(c.variants) > 1),
            "cross_source_entities": len(cross_source),
            "top_entities": [
                {"canonical": c.canonical, "type": c.entity_type, "count": c.count, "sources": len(c.sources)}
                for c in sorted(self.clusters, key=lambda c: c.count, reverse=True)[:20]
            ],
            "type_distribution": dict(type_counts.most_common()),
        }


_entity_index: EntityIndex | None = None


def get_entity_index(rebuild: bool = False) -> EntityIndex:
    """Get or create the global entity index singleton."""
    global _entity_index
    if _entity_index is None or rebuild:
        _entity_index = EntityIndex()
        if not rebuild and _entity_index.load():
            return _entity_index
        _entity_index.build_from_store()
    return _entity_index
