"""Wiki page generation — evidence selection, claim drafting, verification, synthesis.

Pipeline: select_evidence → draft_claims → verify_claims → synthesize_page
Uses cheap model (haiku) for drafting/verification, strong model (sonnet) for synthesis.
Both provider paths (ClaudeProvider + CustomProvider) work via the registry.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

from .config import get_config
from .wiki import WikiPage, WikiManager, get_wiki_manager, _slug, _inputs_hash


# ── Data structures ────────────────────────────────────────────

@dataclass
class EvidenceChunk:
    chunk_id: str
    collection: str
    title: str
    summary: str
    text: str
    importance: int
    keywords: list[str]
    tags: list[str]
    entities: list[str]
    section_heading: str = ""


@dataclass
class Claim:
    claim_id: str
    text: str
    chunk_ids: list[str]
    collections: list[str]
    support_count: int
    status: str = "pending"  # supported | partially_supported | review | conflicted
    verification_note: str = ""
    corroboration: str = "low"

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "chunk_ids": self.chunk_ids,
            "collections": self.collections,
            "support_count": self.support_count,
            "status": self.status,
            "verification_note": self.verification_note,
            "corroboration": self.corroboration,
        }


# ── Evidence Selection ─────────────────────────────────────────

def _parse_list_field(raw) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(v).strip() for v in raw if v]
    if isinstance(raw, str):
        try:
            values = json.loads(raw)
            if isinstance(values, list):
                return [str(v).strip() for v in values if v]
        except (json.JSONDecodeError, ValueError):
            pass
        return [v.strip() for v in raw.split(",") if v.strip()]
    return []


def _parse_entities_field(raw) -> list[str]:
    if not raw:
        return []
    try:
        ents = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(ents, list):
            return [e["name"] for e in ents if isinstance(e, dict) and e.get("name")]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _chunk_to_evidence(chunk: dict) -> EvidenceChunk:
    importance = chunk.get("importance", 3)
    if isinstance(importance, str):
        try:
            importance = int(importance)
        except ValueError:
            importance = 3
    return EvidenceChunk(
        chunk_id=chunk.get("id", ""),
        collection=chunk.get("collection", ""),
        title=chunk.get("title", ""),
        summary=chunk.get("summary", ""),
        text=chunk.get("text", ""),
        importance=importance,
        keywords=_parse_list_field(chunk.get("keywords", "")),
        tags=_parse_list_field(chunk.get("concept_tags", "")),
        entities=_parse_entities_field(chunk.get("entities", "")),
        section_heading=chunk.get("section_heading", ""),
    )


def select_evidence_for_entity(
    entity_name: str,
    max_chunks: int = 30,
    min_chunks: int = 3,
) -> list[EvidenceChunk]:
    """Select evidence chunks for an entity page using postings + graph neighbors."""
    from .cross_index import get_cross_index
    from .entities import get_entity_index
    from .store import get_store

    ci = get_cross_index()
    ei = get_entity_index()
    store = get_store()

    cluster = ei.resolve(entity_name)
    canonical = cluster.canonical if cluster else entity_name

    chunk_ids: set[str] = set()
    for posting_key in [canonical, canonical.lower()]:
        chunk_ids.update(ci.entity_postings.get(posting_key, set()))

    if cluster:
        for variant in cluster.variants:
            chunk_ids.update(ci.entity_postings.get(variant, set()))
            chunk_ids.update(ci.entity_postings.get(variant.lower(), set()))

    evidence: list[EvidenceChunk] = []
    for cid in chunk_ids:
        chunk = store.get_chunk_by_id(cid)
        if chunk:
            evidence.append(_chunk_to_evidence(chunk))

    if len(evidence) < min_chunks:
        return []

    evidence.sort(key=lambda e: (-e.importance, e.chunk_id))
    return evidence[:max_chunks]


def select_evidence_for_concept(
    concept_tag: str,
    max_chunks: int = 30,
    min_chunks: int = 3,
) -> list[EvidenceChunk]:
    """Select evidence chunks for a concept page using postings + keyword graph."""
    from .cross_index import get_cross_index
    from .graph import get_keyword_graph
    from .store import get_store

    ci = get_cross_index()
    store = get_store()

    tag_key = f"tag:{concept_tag.lower()}"
    kw_key = f"kw:{concept_tag.lower()}"

    chunk_ids: set[str] = set()
    chunk_ids.update(ci.term_postings.get(tag_key, set()))
    chunk_ids.update(ci.term_postings.get(kw_key, set()))

    try:
        kg = get_keyword_graph()
        for neighbor in kg.neighbors(concept_tag, n=5):
            term = neighbor["term"]
            chunk_ids.update(ci.term_postings.get(term, set()))
        for member in kg.community_members(concept_tag):
            term = member["term"]
            posting = ci.term_postings.get(term, set())
            if len(posting) <= 50:
                chunk_ids.update(posting)
    except Exception:
        pass

    evidence: list[EvidenceChunk] = []
    for cid in chunk_ids:
        chunk = store.get_chunk_by_id(cid)
        if chunk:
            evidence.append(_chunk_to_evidence(chunk))

    if len(evidence) < min_chunks:
        return []

    collections = Counter(e.collection for e in evidence)
    scored: list[tuple[float, EvidenceChunk]] = []
    for e in evidence:
        tag_match = 1.0 if concept_tag.lower() in [t.lower() for t in e.tags] else 0.3
        diversity_boost = 1.0 / max(collections[e.collection], 1)
        score = tag_match * 2.0 + e.importance * 0.3 + diversity_boost * 0.5
        scored.append((score, e))

    scored.sort(key=lambda x: (-x[0], x[1].chunk_id))
    return [e for _, e in scored[:max_chunks]]


# ── LLM Helpers ────────────────────────────────────────────────

def _get_provider():
    from ..providers.registry import get_registry
    reg = get_registry()
    provider = reg.active
    if not provider:
        raise RuntimeError("No LLM provider available. Configure OpenRouter or Claude CLI.")
    return provider


def _get_model(role: str) -> str | None:
    """Get model name for a pipeline role. Returns None to use provider default."""
    cfg = get_config()
    if role == "draft":
        return cfg.get("wiki.model_draft") or cfg.get("enrichment.model_stage2")
    elif role == "verify":
        return cfg.get("wiki.model_verify") or cfg.get("enrichment.model_stage2")
    elif role == "synthesize":
        return cfg.get("wiki.model_synthesize") or cfg.get("enrichment.model_stage3")
    return None


def _llm_chat(messages: list[dict], role: str = "draft") -> str:
    provider = _get_provider()
    model = _get_model(role)
    if provider.name != "claude":
        model = None
    return provider.chat(messages, model=model)


def _extract_json(text: str) -> dict | list | None:
    """Extract JSON from LLM response, handling markdown fences and prose."""
    text = text.strip()
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        idx_start = text.find(start_char)
        idx_end = text.rfind(end_char)
        if idx_start != -1 and idx_end > idx_start:
            candidate = text[idx_start:idx_end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
    return None


# ── Claim Drafting ─────────────────────────────────────────────

_DRAFT_SYSTEM = """You are a knowledge analyst extracting structured claims from source evidence.

For each claim you identify:
1. Write a clear, specific factual statement grounded in the evidence
2. List the exact chunk_ids that support this claim
3. Note which collections those chunks come from

Output ONLY a JSON array of claim objects:
[
  {
    "text": "Clear factual claim statement",
    "chunk_ids": ["chunk_id_1", "chunk_id_2"],
    "collections": ["source_name"]
  }
]

Rules:
- Each claim must be directly supported by at least one chunk
- Prefer specific claims over vague generalizations
- If multiple sources support a claim, list all supporting chunk_ids
- Keep claims atomic — one idea per claim
- Aim for 5-15 claims depending on evidence richness
- Do NOT invent information beyond what the evidence contains"""


def _format_evidence_for_llm(evidence: list[EvidenceChunk], target_name: str, page_type: str) -> str:
    parts = [f"Extract key claims about '{target_name}' ({page_type}) from the following evidence chunks:\n"]
    for e in evidence:
        parts.append(f"--- chunk_id: {e.chunk_id} | collection: {e.collection} | section: {e.section_heading} ---")
        if e.summary:
            parts.append(f"Summary: {e.summary}")
        text = e.text[:1500] if len(e.text) > 1500 else e.text
        parts.append(f"Text: {text}")
        parts.append("")
    return "\n".join(parts)


def draft_claims(evidence: list[EvidenceChunk], target_name: str, page_type: str) -> list[Claim]:
    """Use cheap model to extract structured claims from evidence."""
    if not evidence:
        return []

    user_prompt = _format_evidence_for_llm(evidence, target_name, page_type)
    response = _llm_chat(
        [{"role": "system", "content": _DRAFT_SYSTEM}, {"role": "user", "content": user_prompt}],
        role="draft",
    )

    raw_claims = _extract_json(response)
    if not isinstance(raw_claims, list):
        print(f"  [wiki_gen] Failed to parse claims JSON for {target_name}")
        return []

    claims: list[Claim] = []
    valid_ids = {e.chunk_id for e in evidence}
    for i, rc in enumerate(raw_claims):
        if not isinstance(rc, dict) or not rc.get("text"):
            continue
        chunk_ids = [cid for cid in rc.get("chunk_ids", []) if cid in valid_ids]
        collections = list(set(rc.get("collections", [])))
        if not chunk_ids:
            chunk_ids = [evidence[0].chunk_id]
            collections = [evidence[0].collection]
        claims.append(Claim(
            claim_id=f"claim-{i:03d}",
            text=rc["text"],
            chunk_ids=chunk_ids,
            collections=collections,
            support_count=len(chunk_ids),
        ))

    print(f"  [wiki_gen] Drafted {len(claims)} claims for {target_name}")
    return claims


# ── Claim Verification ─────────────────────────────────────────

_VERIFY_SYSTEM = """You are a claim verifier. For each claim, check whether the supporting chunk text actually supports what the claim says.

For each claim, output one of:
- "supported": The chunk text directly and clearly supports this claim
- "partially_supported": The chunk text is related but the claim overgeneralizes or adds interpretation
- "review": The chunk text does not clearly support this claim — note what's missing or mismatched
- "conflicted": Different supporting chunks disagree on this claim

Output ONLY a JSON array:
[
  {
    "claim_id": "claim-000",
    "status": "supported",
    "note": ""
  }
]

Be strict but fair. A claim that accurately summarizes the evidence is "supported" even if it paraphrases.
A claim that adds meaning not present in any chunk should be "review" with a note explaining why."""


def verify_claims(
    claims: list[Claim],
    evidence: list[EvidenceChunk],
) -> list[Claim]:
    """Use cheap model to verify each claim against its supporting chunk text."""
    if not claims:
        return []

    evidence_map = {e.chunk_id: e for e in evidence}

    parts = ["Verify each claim against its supporting evidence:\n"]
    for claim in claims:
        parts.append(f"## {claim.claim_id}")
        parts.append(f"Claim: {claim.text}")
        parts.append(f"Supporting chunks:")
        for cid in claim.chunk_ids:
            ev = evidence_map.get(cid)
            if ev:
                text = ev.text[:800] if len(ev.text) > 800 else ev.text
                parts.append(f"  [{cid}]: {text}")
        parts.append("")

    response = _llm_chat(
        [{"role": "system", "content": _VERIFY_SYSTEM}, {"role": "user", "content": "\n".join(parts)}],
        role="verify",
    )

    verdicts = _extract_json(response)
    if isinstance(verdicts, list):
        verdict_map = {v["claim_id"]: v for v in verdicts if isinstance(v, dict) and v.get("claim_id")}
    else:
        print(f"  [wiki_gen] Failed to parse verification JSON, marking all as review")
        verdict_map = {}

    for claim in claims:
        v = verdict_map.get(claim.claim_id, {})
        claim.status = v.get("status", "review")
        claim.verification_note = v.get("note", "verification parse failure" if not verdict_map else "")
        if claim.status not in ("supported", "partially_supported", "review", "conflicted"):
            claim.status = "review"
        claim.corroboration = _compute_corroboration(claim)

    verified = sum(1 for c in claims if c.status == "supported")
    partial = sum(1 for c in claims if c.status == "partially_supported")
    review = sum(1 for c in claims if c.status == "review")
    conflicted = sum(1 for c in claims if c.status == "conflicted")
    print(f"  [wiki_gen] Verified: {verified} supported, {partial} partial, {review} review, {conflicted} conflicted")
    return claims


def _compute_corroboration(claim: Claim) -> str:
    n_chunks = claim.support_count
    n_colls = len(set(claim.collections))
    if claim.status == "conflicted":
        return "mixed"
    if n_chunks >= 3 and n_colls >= 2:
        return "high"
    if n_chunks >= 2:
        return "moderate"
    return "low"


# ── Page Synthesis ─────────────────────────────────────────────

_SYNTHESIZE_SYSTEM = """You are a wiki page writer for a knowledge base. Write a clear, well-structured wiki page from verified claims.

Page structure:
# {title}

## Summary
2-4 sentence overview synthesizing the key points.

## Key Claims
For each claim, write it as a bullet point. After each claim, on the next line write:
  Sources: `chunk_id_1`, `chunk_id_2`
  Support: N chunks, M collections — STATUS

## Cross-Source Synthesis
If claims come from multiple sources, write 1-3 paragraphs about what the sources collectively show. If single-source, write about the depth and coverage of the source's treatment.

## Tensions
If any claims are conflicted or partially_supported, discuss the tensions. If none, write "No significant tensions identified across the evidence."

## Related Pages
List related wiki pages as [[page_type/slug]] links.

## Provenance
List all chunk_ids that contributed to this page.

Rules:
- Write in a neutral, encyclopedic tone
- Every claim must reference its supporting chunks
- Do not invent information beyond the verified claims
- Claims marked "review" should be included but noted as needing verification
- Be concise — this is a reference page, not an essay"""


def synthesize_page(
    target_name: str,
    page_type: str,
    claims: list[Claim],
    evidence: list[EvidenceChunk],
    related_pages: list[str],
) -> str:
    """Use strong model to write the final wiki page from verified claims."""
    claims_text = []
    for c in claims:
        claims_text.append(f"- claim_id: {c.claim_id}")
        claims_text.append(f"  text: {c.text}")
        claims_text.append(f"  status: {c.status}")
        claims_text.append(f"  chunk_ids: {', '.join(c.chunk_ids)}")
        claims_text.append(f"  collections: {', '.join(c.collections)}")
        claims_text.append(f"  corroboration: {c.corroboration}")
        if c.verification_note:
            claims_text.append(f"  note: {c.verification_note}")
        claims_text.append("")

    collections_involved = sorted(set(c for claim in claims for c in claim.collections))
    related_text = "\n".join(f"  - [[{rp}]]" for rp in related_pages) if related_pages else "  (none identified)"

    user_prompt = f"""Write a wiki page for the {page_type}: "{target_name}"

Verified claims:
{chr(10).join(claims_text)}

Collections involved: {', '.join(collections_involved)}
Related pages to link:
{related_text}

Write the complete page now."""

    return _llm_chat(
        [{"role": "system", "content": _SYNTHESIZE_SYSTEM}, {"role": "user", "content": user_prompt}],
        role="synthesize",
    )


# ── Orchestrator ───────────────────────────────────────────────

def _find_related_pages(target_name: str, page_type: str, evidence: list[EvidenceChunk]) -> list[str]:
    """Determine related wiki pages from evidence metadata."""
    wm = get_wiki_manager()
    existing = {p["page_id"] for p in wm.list_pages()}

    related: set[str] = set()
    all_tags = set()
    all_entities = set()
    all_collections = set()

    for e in evidence:
        for t in e.tags:
            all_tags.add(t.lower())
        for ent in e.entities:
            all_entities.add(ent.lower())
        all_collections.add(e.collection)

    if page_type == "entity":
        for tag in all_tags:
            candidate = f"concept/{_slug(tag)}"
            if candidate in existing:
                related.add(candidate)
    elif page_type == "concept":
        for ent in all_entities:
            candidate = f"entity/{_slug(ent)}"
            if candidate in existing:
                related.add(candidate)

    for coll in all_collections:
        candidate = f"source/{_slug(coll.replace('_', ' '))}"
        if candidate in existing:
            related.add(candidate)

    own_id = f"{page_type}/{_slug(target_name)}"
    related.discard(own_id)
    return sorted(related)


def generate_entity_page(
    entity_name: str,
    force: bool = False,
    max_chunks: int = 30,
) -> WikiPage | None:
    """Generate or refresh a wiki page for an entity."""
    from .entities import get_entity_index

    ei = get_entity_index()
    cluster = ei.resolve(entity_name)
    canonical = cluster.canonical if cluster else entity_name
    slug = _slug(canonical)
    page_id = f"entity/{slug}"

    wm = get_wiki_manager()
    if not force and wm.page_exists(page_id):
        existing = wm.get_page(page_id)
        if existing and existing.status != "stale":
            print(f"  [wiki_gen] Page already exists: {page_id}")
            return existing

    print(f"  [wiki_gen] Generating entity page: {canonical}")
    evidence = select_evidence_for_entity(canonical, max_chunks=max_chunks)
    if not evidence:
        print(f"  [wiki_gen] Not enough evidence for entity: {canonical}")
        return None

    print(f"  [wiki_gen] {len(evidence)} evidence chunks from {len(set(e.collection for e in evidence))} collections")

    claims = draft_claims(evidence, canonical, "entity")
    if not claims:
        return None

    claims = verify_claims(claims, evidence)
    related = _find_related_pages(canonical, "entity", evidence)
    content = synthesize_page(canonical, "entity", claims, evidence, related)

    collections = sorted(set(e.collection for e in evidence))
    chunk_ids = sorted(set(e.chunk_id for e in evidence))

    page = WikiPage(
        page_id=page_id,
        page_type="entity",
        title=canonical,
        slug=slug,
        version=(wm.get_page(page_id).version + 1) if wm.page_exists(page_id) else 1,
        source_collections=collections,
        source_chunk_count=len(chunk_ids),
        supporting_source_count=len(collections),
        corroboration_level=_page_corroboration(claims),
        confidence=_page_confidence(claims),
        generation={
            "strategy": "synthesized",
            "model_draft": _get_model("draft") or "default",
            "model_verify": _get_model("verify") or "default",
            "model_synthesize": _get_model("synthesize") or "default",
            "inputs_hash": _inputs_hash(chunk_ids),
            "source_versions": {},
            "claim_count": len(claims),
            "claims": [c.to_dict() for c in claims],
        },
        canonical_sources=[{"collection": c, "weight": "primary" if i == 0 else "supporting"} for i, c in enumerate(collections)],
        related_pages=related,
        content=content,
    )

    wm.save_page(page)
    print(f"  [wiki_gen] Saved entity page: {page_id} ({len(claims)} claims)")
    return page


def generate_concept_page(
    concept_tag: str,
    force: bool = False,
    max_chunks: int = 30,
) -> WikiPage | None:
    """Generate or refresh a wiki page for a concept."""
    slug = _slug(concept_tag)
    page_id = f"concept/{slug}"

    wm = get_wiki_manager()
    if not force and wm.page_exists(page_id):
        existing = wm.get_page(page_id)
        if existing and existing.status != "stale":
            print(f"  [wiki_gen] Page already exists: {page_id}")
            return existing

    print(f"  [wiki_gen] Generating concept page: {concept_tag}")
    evidence = select_evidence_for_concept(concept_tag, max_chunks=max_chunks)
    if not evidence:
        print(f"  [wiki_gen] Not enough evidence for concept: {concept_tag}")
        return None

    print(f"  [wiki_gen] {len(evidence)} evidence chunks from {len(set(e.collection for e in evidence))} collections")

    claims = draft_claims(evidence, concept_tag, "concept")
    if not claims:
        return None

    claims = verify_claims(claims, evidence)
    related = _find_related_pages(concept_tag, "concept", evidence)
    content = synthesize_page(concept_tag, "concept", claims, evidence, related)

    collections = sorted(set(e.collection for e in evidence))
    chunk_ids = sorted(set(e.chunk_id for e in evidence))

    page = WikiPage(
        page_id=page_id,
        page_type="concept",
        title=concept_tag.replace("-", " ").title(),
        slug=slug,
        version=(wm.get_page(page_id).version + 1) if wm.page_exists(page_id) else 1,
        source_collections=collections,
        source_chunk_count=len(chunk_ids),
        supporting_source_count=len(collections),
        corroboration_level=_page_corroboration(claims),
        confidence=_page_confidence(claims),
        generation={
            "strategy": "synthesized",
            "model_draft": _get_model("draft") or "default",
            "model_verify": _get_model("verify") or "default",
            "model_synthesize": _get_model("synthesize") or "default",
            "inputs_hash": _inputs_hash(chunk_ids),
            "source_versions": {},
            "claim_count": len(claims),
            "claims": [c.to_dict() for c in claims],
        },
        canonical_sources=[{"collection": c, "weight": "primary" if i == 0 else "supporting"} for i, c in enumerate(collections)],
        related_pages=related,
        content=content,
    )

    wm.save_page(page)
    print(f"  [wiki_gen] Saved concept page: {page_id} ({len(claims)} claims)")
    return page


def _page_corroboration(claims: list[Claim]) -> str:
    if not claims:
        return "low"
    levels = [c.corroboration for c in claims if c.status in ("supported", "partially_supported")]
    if not levels:
        return "low"
    if any(c.status == "conflicted" for c in claims):
        return "mixed"
    if sum(1 for l in levels if l == "high") > len(levels) / 2:
        return "high"
    if sum(1 for l in levels if l in ("high", "moderate")) > len(levels) / 2:
        return "moderate"
    return "low"


def _page_confidence(claims: list[Claim]) -> str:
    if not claims:
        return "low"
    supported = sum(1 for c in claims if c.status == "supported")
    total = len(claims)
    if supported / total >= 0.8:
        return "high"
    if supported / total >= 0.5:
        return "medium"
    return "low"


# ── Batch Generation ───────────────────────────────────────────

def generate_wiki_pages(
    page_type: str = "concept",
    limit: int = 10,
    force: bool = False,
    min_chunks: int = 3,
) -> list[WikiPage]:
    """Generate multiple wiki pages for top candidates of a given type."""
    pages: list[WikiPage] = []

    if page_type == "concept":
        candidates = _get_concept_candidates(min_chunks=min_chunks)
        print(f"  [wiki_gen] {len(candidates)} concept candidates, generating top {limit}")
        for tag, count, n_colls in candidates[:limit]:
            try:
                page = generate_concept_page(tag, force=force)
                if page:
                    pages.append(page)
            except Exception as e:
                print(f"  [wiki_gen] Failed to generate concept/{_slug(tag)}: {e}")

    elif page_type == "entity":
        candidates = _get_entity_candidates(min_mentions=2)
        print(f"  [wiki_gen] {len(candidates)} entity candidates, generating top {limit}")
        for name, count, n_colls in candidates[:limit]:
            try:
                page = generate_entity_page(name, force=force)
                if page:
                    pages.append(page)
            except Exception as e:
                print(f"  [wiki_gen] Failed to generate entity/{_slug(name)}: {e}")

    print(f"  [wiki_gen] Generated {len(pages)} {page_type} pages")
    return pages


def _get_concept_candidates(min_chunks: int = 3) -> list[tuple[str, int, int]]:
    from .cross_index import get_cross_index
    ci = get_cross_index()

    tag_counts: Counter[str] = Counter()
    tag_collections: dict[str, set[str]] = {}
    for cf in ci.by_chunk.values():
        for t in cf.tags:
            tag_counts[t] += 1
            tag_collections.setdefault(t, set()).add(cf.collection)

    candidates = []
    for tag, count in tag_counts.items():
        n_colls = len(tag_collections[tag])
        if count >= min_chunks or n_colls >= 2:
            candidates.append((tag, count, n_colls))

    candidates.sort(key=lambda x: (-x[2], -x[1]))
    return candidates


def _get_entity_candidates(min_mentions: int = 2) -> list[tuple[str, int, int]]:
    from .entities import get_entity_index
    ei = get_entity_index()

    candidates = []
    for cluster in ei.clusters:
        if cluster.count >= min_mentions or len(cluster.sources) >= 2:
            candidates.append((cluster.canonical, cluster.count, len(cluster.sources)))

    candidates.sort(key=lambda x: (-x[2], -x[1]))
    return candidates
