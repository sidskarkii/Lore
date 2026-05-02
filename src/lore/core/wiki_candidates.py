"""Wiki candidate discovery and ranking for recursive generation.

Discovers missing wiki pages from broken links, EntityIndex gaps, and
CrossSourceIndex term gaps. Ranks by link pressure, evidence count,
source diversity, and graph centrality. Returns scored candidates
with cost estimates.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Candidate:
    page_type: str  # entity or concept
    target: str
    slug: str
    link_pressure: int = 0
    evidence_count: int = 0
    source_count: int = 0
    centrality: float = 0.0
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "page_type": self.page_type,
            "target": self.target,
            "slug": self.slug,
            "link_pressure": self.link_pressure,
            "evidence_count": self.evidence_count,
            "source_count": self.source_count,
            "centrality": round(self.centrality, 4),
            "score": round(self.score, 4),
        }


def discover_candidates(
    repair_only: bool = False,
    min_chunks_concept: int = 3,
    min_chunks_entity: int = 2,
) -> list[Candidate]:
    """Discover all missing wiki page candidates, scored and ranked."""
    from .wiki import get_wiki_manager, _slug

    wm = get_wiki_manager()
    existing = {m["page_id"] for m in wm.list_pages()}

    link_pressure = _compute_link_pressure(wm)
    candidates: dict[str, Candidate] = {}

    for page_id, pressure in link_pressure.items():
        if page_id in existing:
            continue
        ptype, slug = page_id.split("/", 1) if "/" in page_id else ("concept", page_id)
        if ptype not in ("entity", "concept"):
            continue
        candidates[page_id] = Candidate(
            page_type=ptype, target=slug, slug=slug,
            link_pressure=pressure,
        )

    if not repair_only:
        _add_entity_candidates(candidates, existing, min_chunks_entity)
        _add_concept_candidates(candidates, existing, min_chunks_concept)

    _enrich_evidence(candidates)
    _enrich_centrality(candidates)
    _compute_scores(candidates)

    ranked = sorted(candidates.values(), key=lambda c: -c.score)
    return ranked


def plan(
    repair_only: bool = False,
    limit: int = 25,
    min_chunks_concept: int = 3,
    min_chunks_entity: int = 2,
) -> dict:
    """Dry-run: return ranked candidates with cost estimates."""
    candidates = discover_candidates(
        repair_only=repair_only,
        min_chunks_concept=min_chunks_concept,
        min_chunks_entity=min_chunks_entity,
    )

    capped = candidates[:limit]
    entity_count = sum(1 for c in capped if c.page_type == "entity")
    concept_count = sum(1 for c in capped if c.page_type == "concept")
    llm_calls = len(capped) * 4

    return {
        "total_candidates": len(candidates),
        "planned": len(capped),
        "entity_pages": entity_count,
        "concept_pages": concept_count,
        "estimated_llm_calls": llm_calls,
        "estimated_batches": max(1, (len(capped) + 24) // 25),
        "candidates": [c.to_dict() for c in capped],
    }


def _compute_link_pressure(wm) -> dict[str, int]:
    """Count how many existing pages link to each missing page."""
    pressure: Counter[str] = Counter()
    existing = {m["page_id"] for m in wm.list_pages()}
    for meta in wm.list_pages():
        page = wm.get_page(meta["page_id"])
        if not page:
            continue
        for related in page.related_pages:
            if related not in existing:
                pressure[related] += 1
    return dict(pressure)


def _add_entity_candidates(
    candidates: dict[str, Candidate],
    existing: set[str],
    min_mentions: int,
):
    from .entities import get_entity_index
    from .wiki import _slug

    ei = get_entity_index()
    for cluster in ei.clusters:
        if cluster.count < min_mentions and len(cluster.sources) < 2:
            continue
        slug = _slug(cluster.canonical)
        page_id = f"entity/{slug}"
        if page_id in existing or not slug:
            continue
        if page_id in candidates:
            candidates[page_id].evidence_count = cluster.count
            candidates[page_id].source_count = len(cluster.sources)
        else:
            candidates[page_id] = Candidate(
                page_type="entity", target=cluster.canonical, slug=slug,
                evidence_count=cluster.count,
                source_count=len(cluster.sources),
            )


def _add_concept_candidates(
    candidates: dict[str, Candidate],
    existing: set[str],
    min_chunks: int,
):
    from .cross_index import get_cross_index
    from .wiki import _slug

    ci = get_cross_index()
    tag_counts: Counter[str] = Counter()
    tag_collections: dict[str, set[str]] = {}
    for cf in ci.by_chunk.values():
        for t in cf.tags:
            tag_counts[t] += 1
            tag_collections.setdefault(t, set()).add(cf.collection)

    for tag, count in tag_counts.items():
        n_colls = len(tag_collections.get(tag, set()))
        if count < min_chunks and n_colls < 2:
            continue
        slug = _slug(tag.split(":", 1)[-1] if ":" in tag else tag)
        page_id = f"concept/{slug}"
        if page_id in existing or not slug:
            continue
        if page_id in candidates:
            c = candidates[page_id]
            c.evidence_count = max(c.evidence_count, count)
            c.source_count = max(c.source_count, n_colls)
        else:
            candidates[page_id] = Candidate(
                page_type="concept", target=tag, slug=slug,
                evidence_count=count,
                source_count=n_colls,
            )


def _enrich_evidence(candidates: dict[str, Candidate]):
    """Fill in evidence_count from postings for repair-only candidates that lack it."""
    from .cross_index import get_cross_index
    from .entities import get_entity_index

    ci = get_cross_index()
    try:
        ei = get_entity_index()
    except Exception:
        ei = None

    for page_id, c in candidates.items():
        if c.evidence_count > 0:
            continue
        if c.page_type == "entity" and ei:
            cluster = ei.resolve(c.target)
            if cluster:
                c.evidence_count = cluster.count
                c.source_count = len(cluster.sources)
        elif c.page_type == "concept":
            tag_key = f"tag:{c.target.lower()}"
            kw_key = f"kw:{c.target.lower()}"
            chunk_ids = ci.term_postings.get(tag_key, set()) | ci.term_postings.get(kw_key, set())
            c.evidence_count = len(chunk_ids)
            colls = set()
            for cf in [ci.by_chunk.get(cid) for cid in chunk_ids]:
                if cf:
                    colls.add(cf.collection)
            c.source_count = len(colls)


def _enrich_centrality(candidates: dict[str, Candidate]):
    """Add graph centrality scores where available."""
    try:
        from .graph import get_entity_graph, get_keyword_graph
        eg = get_entity_graph()
        kg = get_keyword_graph()
    except Exception:
        return

    for c in candidates.values():
        if c.page_type == "entity":
            try:
                neighbors = eg.neighbors(c.target, n=1)
                if neighbors:
                    c.centrality = neighbors[0].get("npmi", 0)
            except Exception:
                pass
        elif c.page_type == "concept":
            try:
                neighbors = kg.neighbors(c.target, n=1)
                if neighbors:
                    c.centrality = neighbors[0].get("npmi", 0)
            except Exception:
                pass


def _compute_scores(candidates: dict[str, Candidate]):
    """Weighted normalized scoring: link_pressure, evidence, source_diversity, centrality."""
    if not candidates:
        return

    vals = list(candidates.values())
    max_lp = max((c.link_pressure for c in vals), default=1) or 1
    max_ev = max((c.evidence_count for c in vals), default=1) or 1
    max_src = max((c.source_count for c in vals), default=1) or 1
    max_cent = max((c.centrality for c in vals), default=1) or 1

    for c in vals:
        lp_norm = c.link_pressure / max_lp
        ev_norm = c.evidence_count / max_ev
        src_norm = c.source_count / max_src
        cent_norm = c.centrality / max_cent if max_cent > 0 else 0

        c.score = (
            0.40 * lp_norm
            + 0.25 * ev_norm
            + 0.20 * src_norm
            + 0.15 * cent_norm
        )
