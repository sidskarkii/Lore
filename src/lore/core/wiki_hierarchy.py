"""Wiki page hierarchy — topic clustering and index surfaces.

Clusters pages into topic groups using shared concept/entity references
(Jaccard similarity on related_pages). Generates a hierarchy manifest
with type indexes and topic clusters. No LLM calls.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def build_hierarchy() -> dict:
    """Build the full page hierarchy: type indexes + topic clusters."""
    from .wiki import get_wiki_manager

    wm = get_wiki_manager()
    pages = wm.list_pages()

    type_indexes = _build_type_indexes(pages, wm)
    clusters = _cluster_pages(pages, wm)
    hierarchy = {
        "type_indexes": type_indexes,
        "topic_clusters": [c.to_dict() for c in clusters],
        "total_pages": len(pages),
        "total_clusters": len(clusters),
    }

    _save_hierarchy(hierarchy)
    return hierarchy


class TopicCluster:
    def __init__(self, cluster_id: int, label: str):
        self.cluster_id = cluster_id
        self.label = label
        self.pages: list[dict] = []
        self.key_terms: list[str] = []

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "key_terms": self.key_terms[:10],
            "page_count": len(self.pages),
            "pages": self.pages,
        }


def _build_type_indexes(pages: list[dict], wm) -> dict[str, list[dict]]:
    indexes: dict[str, list[dict]] = {}
    for meta in pages:
        ptype = meta.get("page_type", "unknown")
        entry = {
            "page_id": meta["page_id"],
            "title": meta.get("title", ""),
            "status": meta.get("status", ""),
            "source_collections": meta.get("source_collections", []),
            "supporting_source_count": meta.get("supporting_source_count", 0),
        }
        indexes.setdefault(ptype, []).append(entry)

    for ptype in indexes:
        indexes[ptype].sort(key=lambda p: p["title"].lower())

    return indexes


def _cluster_pages(pages: list[dict], wm) -> list[TopicCluster]:
    """Cluster pages by shared concept/entity references using average-linkage Jaccard."""
    page_tags: dict[str, set[str]] = {}
    for meta in pages:
        page = wm.get_page(meta["page_id"])
        if not page:
            continue
        tags = set()
        for r in page.related_pages:
            if r.startswith("concept/") or r.startswith("entity/"):
                tags.add(r)
        page_tags[meta["page_id"]] = tags

    if len(page_tags) < 2:
        if page_tags:
            pid = list(page_tags.keys())[0]
            c = TopicCluster(0, _label_from_tags(page_tags[pid]))
            meta = next((m for m in pages if m["page_id"] == pid), {})
            c.pages.append({"page_id": pid, "title": meta.get("title", "")})
            return [c]
        return []

    clusters = _average_linkage_cluster(page_tags, threshold=0.20)

    result = []
    for i, group in enumerate(clusters):
        tag_counts: Counter[str] = Counter()
        for pid in group:
            for t in page_tags.get(pid, set()):
                tag_counts[t] += 1

        shared_tags = {t for t, count in tag_counts.items() if count >= 2} if len(group) >= 2 else set(tag_counts)
        label = _label_from_tags(shared_tags if shared_tags else set(tag_counts))
        cluster = TopicCluster(i, label)
        cluster.key_terms = [
            t for t, _ in tag_counts.most_common(10)
            if t.startswith("concept/") or t.startswith("entity/")
        ]

        for pid in sorted(group):
            meta = next((m for m in pages if m["page_id"] == pid), {})
            cluster.pages.append({
                "page_id": pid,
                "title": meta.get("title", ""),
                "page_type": meta.get("page_type", ""),
            })
        result.append(cluster)

    result.sort(key=lambda c: -len(c.pages))
    return result


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _average_linkage_cluster(
    page_tags: dict[str, set[str]],
    threshold: float = 0.20,
) -> list[list[str]]:
    """Average-linkage clustering on Jaccard similarity."""
    pids = sorted(page_tags.keys())
    clusters: list[list[str]] = []

    for pid in pids:
        best_cluster = -1
        best_avg = 0.0

        for ci, group in enumerate(clusters):
            total = sum(_jaccard(page_tags[pid], page_tags[m]) for m in group)
            avg = total / len(group)
            if avg > best_avg:
                best_avg = avg
                best_cluster = ci

        if best_avg >= threshold and best_cluster >= 0:
            clusters[best_cluster].append(pid)
        else:
            clusters.append([pid])

    return clusters


def _label_from_tags(tags: set[str]) -> str:
    concepts = []
    for t in sorted(tags):
        if t.startswith("concept/"):
            name = t.split("/", 1)[1].replace("-", " ").title()
            concepts.append(name)
        elif t.startswith("entity/"):
            name = t.split("/", 1)[1].replace("-", " ").title()
            concepts.append(name)
    if concepts:
        return ", ".join(concepts[:3])
    return "Miscellaneous"


def _save_hierarchy(hierarchy: dict):
    from .config import get_config
    cfg = get_config()
    path = cfg.data_dir / "wiki" / "manifests" / "hierarchy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hierarchy, indent=2))


def load_hierarchy() -> dict | None:
    from .config import get_config
    cfg = get_config()
    path = cfg.data_dir / "wiki" / "manifests" / "hierarchy.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
