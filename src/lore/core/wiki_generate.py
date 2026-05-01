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

    collections = Counter(e.collection for e in evidence)
    n_colls = len(collections)
    per_coll_cap = max(max_chunks // max(n_colls, 1), 5)

    evidence.sort(key=lambda e: (-e.importance, e.chunk_id))
    selected: list[EvidenceChunk] = []
    coll_counts: Counter[str] = Counter()
    overflow: list[EvidenceChunk] = []
    for e in evidence:
        if coll_counts[e.collection] < per_coll_cap:
            selected.append(e)
            coll_counts[e.collection] += 1
        else:
            overflow.append(e)
        if len(selected) >= max_chunks:
            break

    for e in overflow:
        if len(selected) >= max_chunks:
            break
        selected.append(e)

    return selected


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


_CLAUDE_SHORTHANDS = {"haiku", "sonnet", "opus"}


def _llm_chat(messages: list[dict], role: str = "draft") -> str:
    provider = _get_provider()
    model = _get_model(role)
    if provider.name != "claude" and model in _CLAUDE_SHORTHANDS:
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


_MAX_DRAFT_CHARS = 60_000
_MAX_VERIFY_CHARS = 40_000


def _format_evidence_for_llm(evidence: list[EvidenceChunk], target_name: str, page_type: str) -> str:
    parts = [f"Extract key claims about '{target_name}' ({page_type}) from the following evidence chunks:\n"]
    budget = _MAX_DRAFT_CHARS
    for e in evidence:
        header = f"--- chunk_id: {e.chunk_id} | collection: {e.collection} | section: {e.section_heading} ---"
        summary_line = f"Summary: {e.summary}" if e.summary else ""
        text = e.text[:1500] if len(e.text) > 1500 else e.text
        block = f"{header}\n{summary_line}\nText: {text}\n"
        if budget - len(block) < 0:
            parts.append(f"(... {len(evidence) - len(parts) + 1} chunks truncated for token budget)")
            break
        parts.append(block)
        budget -= len(block)
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
        status = "pending"
        note = ""
        if not chunk_ids:
            status = "review"
            note = "drafter returned no valid chunk_ids — claim has no provenance"
            chunk_ids = []
            collections = []
        claims.append(Claim(
            claim_id=f"claim-{i:03d}",
            text=rc["text"],
            chunk_ids=chunk_ids,
            collections=collections,
            support_count=len(chunk_ids),
            status=status,
            verification_note=note,
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

    verifiable = [c for c in claims if c.status != "review"]
    if not verifiable:
        return claims

    parts = ["Verify each claim against its supporting evidence:\n"]
    budget = _MAX_VERIFY_CHARS
    for claim in verifiable:
        block_parts = [f"## {claim.claim_id}", f"Claim: {claim.text}", "Supporting chunks:"]
        for cid in claim.chunk_ids:
            ev = evidence_map.get(cid)
            if ev:
                text = ev.text[:800] if len(ev.text) > 800 else ev.text
                block_parts.append(f"  [{cid}]: {text}")
        block = "\n".join(block_parts)
        if budget - len(block) < 0:
            break
        parts.append(block)
        budget -= len(block)
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

    for claim in verifiable:
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
    from .entities import get_entity_index

    wm = get_wiki_manager()
    ei = get_entity_index()
    existing = {p["page_id"] for p in wm.list_pages()}

    related: set[str] = set()
    all_tags = set()
    all_entities = set()
    all_collections = set()

    for e in evidence:
        for t in e.tags:
            all_tags.add(t.lower())
        for ent in e.entities:
            cluster = ei.resolve(ent)
            canonical = cluster.canonical if cluster else ent
            all_entities.add(canonical)
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


# ── Comparison Pages ──────────────────────────────────────────

_MIN_EVIDENCE_PER_COLLECTION = 2
_MAX_EVIDENCE_PER_COLLECTION = 8
_MAX_COMPARISON_COLLECTIONS = 4


def select_evidence_for_comparison(
    topic: str,
    collections: list[str],
    max_per_collection: int = _MAX_EVIDENCE_PER_COLLECTION,
    min_per_collection: int = _MIN_EVIDENCE_PER_COLLECTION,
) -> dict[str, list[EvidenceChunk]]:
    """Select evidence per collection using semantic search + postings expansion."""
    from .search import SearchEngine
    from .cross_index import get_cross_index

    engine = SearchEngine()
    ci = get_cross_index()

    evidence_by_coll: dict[str, list[EvidenceChunk]] = {}

    for coll in collections:
        results = engine.search(
            query=topic,
            n_results=max_per_collection * 2,
            collection=coll,
            expand=False,
            _skip_expansion=True,
        )

        chunk_ids = {r["id"] for r in results if r.get("id")}

        tag_key = f"tag:{topic.lower()}"
        kw_key = f"kw:{topic.lower()}"
        posting_ids = set()
        posting_ids.update(ci.term_postings.get(tag_key, set()))
        posting_ids.update(ci.term_postings.get(kw_key, set()))
        for pid in posting_ids:
            coll_meta = ci.by_chunk.get(pid)
            if coll_meta and coll_meta.collection == coll:
                chunk_ids.add(pid)

        from .store import get_store
        store = get_store()
        chunks = []
        for cid in chunk_ids:
            chunk = store.get_chunk_by_id(cid)
            if chunk:
                chunks.append(_chunk_to_evidence(chunk))

        chunks.sort(key=lambda e: (-e.importance, e.chunk_id))
        chunks = chunks[:max_per_collection]

        if len(chunks) >= min_per_collection:
            evidence_by_coll[coll] = chunks

    return evidence_by_coll


_DISTILL_SYSTEM = """You are a knowledge analyst distilling one source's position on a topic.

Given evidence chunks from a single source about a specific topic, extract:
1. The source's key positions/claims on this topic
2. Unique perspectives or frameworks the source offers
3. Specific examples or evidence the source provides

Output ONLY a JSON object:
{
  "positions": [
    {
      "text": "Clear statement of the source's position",
      "chunk_ids": ["chunk_id_1", "chunk_id_2"],
      "strength": "strong|moderate|mentioned"
    }
  ],
  "unique_angle": "What makes this source's treatment distinctive (1-2 sentences)",
  "coverage": "deep|moderate|shallow"
}

Rules:
- Ground every position in specific chunk_ids
- "strong" = central argument, "moderate" = discussed, "mentioned" = brief reference
- Be faithful to what the source actually says, do not extrapolate"""


def _distill_source_position(
    topic: str,
    collection: str,
    evidence: list[EvidenceChunk],
) -> dict | None:
    """Distill a single source's position on a topic (haiku)."""
    parts = [f"Distill {collection.replace('_', ' ')}'s position on '{topic}' from these chunks:\n"]
    budget = _MAX_DRAFT_CHARS
    for e in evidence:
        header = f"--- chunk_id: {e.chunk_id} | section: {e.section_heading} ---"
        text = e.text[:1500] if len(e.text) > 1500 else e.text
        block = f"{header}\n{text}\n"
        if budget - len(block) < 0:
            break
        parts.append(block)
        budget -= len(block)

    response = _llm_chat(
        [{"role": "system", "content": _DISTILL_SYSTEM}, {"role": "user", "content": "\n".join(parts)}],
        role="draft",
    )

    result = _extract_json(response)
    if not isinstance(result, dict):
        return None

    valid_ids = {e.chunk_id for e in evidence}
    for pos in result.get("positions", []):
        pos["chunk_ids"] = [cid for cid in pos.get("chunk_ids", []) if cid in valid_ids]
        pos["collection"] = collection
    result["collection"] = collection

    return result


_COMPARE_SYSTEM = """You are a comparative analyst synthesizing multiple sources' positions on a topic.

Given distilled positions from {n_sources} sources, produce a structured comparison.

Output ONLY a JSON object:
{{
  "agreements": [
    {{
      "text": "Point where sources agree",
      "sources": ["collection_a", "collection_b"],
      "chunk_ids": ["id1", "id2"]
    }}
  ],
  "tensions": [
    {{
      "text": "Point where sources disagree or differ",
      "source_positions": {{
        "collection_a": "Their position",
        "collection_b": "Their contrasting position"
      }},
      "chunk_ids": ["id1", "id2"]
    }}
  ],
  "unique_contributions": [
    {{
      "collection": "collection_a",
      "text": "What this source uniquely adds",
      "chunk_ids": ["id1"]
    }}
  ],
  "synthesis": "2-3 sentence overall synthesis of how these sources relate on this topic"
}}

Rules:
- Only include agreements where 2+ sources genuinely converge
- Tensions should reflect real differences, not just different emphasis
- Unique contributions = perspectives only one source offers
- Include chunk_ids for every claim
- If a section has no items, use an empty array — do not force balance"""


_COMPARISON_SYNTH_SYSTEM = """You are a wiki page writer creating a comparison page for a knowledge base.

Write a structured comparison page from analyzed source positions.

Page structure:
# {{title}}

## Summary
2-4 sentence overview of how these sources relate on this topic.

## Per-Source Positions
### {{Source A}}
Key positions from this source, with chunk citations.
### {{Source B}}
Key positions from this source, with chunk citations.

## Agreements
Points where sources converge. If none, write "No significant agreements identified."

## Tensions & Disagreements
Points where sources differ. If none, write "No significant tensions identified."

## Unique Contributions
Perspectives that only one source offers.

## Synthesis
1-3 paragraphs synthesizing the overall picture across sources.

## Provenance
List all contributing chunk_ids.

Rules:
- Neutral, encyclopedic tone
- Every claim references chunk_ids
- Do not invent beyond what the analysis contains
- Allow sparse sections — not every comparison has tensions or agreements"""


def _compare_positions(
    topic: str,
    distilled: list[dict],
) -> dict | None:
    """Cross-source comparison from distilled positions (sonnet)."""
    parts = [f"Compare these {len(distilled)} sources on '{topic}':\n"]
    for d in distilled:
        coll = d.get("collection", "unknown")
        parts.append(f"\n## {coll.replace('_', ' ')}")
        parts.append(f"Unique angle: {d.get('unique_angle', 'N/A')}")
        parts.append(f"Coverage: {d.get('coverage', 'N/A')}")
        parts.append("Positions:")
        for pos in d.get("positions", []):
            strength = pos.get("strength", "moderate")
            parts.append(f"  [{strength}] {pos.get('text', '')}")
            parts.append(f"    Chunks: {', '.join(pos.get('chunk_ids', []))}")

    prompt = _COMPARE_SYSTEM.format(n_sources=len(distilled))
    response = _llm_chat(
        [{"role": "system", "content": prompt}, {"role": "user", "content": "\n".join(parts)}],
        role="synthesize",
    )

    return _extract_json(response)


def _synthesize_comparison(
    topic: str,
    collections: list[str],
    distilled: list[dict],
    comparison: dict,
    related_pages: list[str],
) -> str:
    """Synthesize the final comparison page markdown (sonnet)."""
    coll_display = " vs ".join(c.replace("_", " ") for c in collections)
    title = f"{topic.replace('-', ' ').title()}: {coll_display}"

    related_text = "\n".join(f"  - [[{rp}]]" for rp in related_pages) if related_pages else "  (none)"

    user_prompt = f"""Write a comparison wiki page: "{title}"

Per-source distillations:
{json.dumps(distilled, indent=2)}

Cross-source analysis:
{json.dumps(comparison, indent=2)}

Collections being compared: {', '.join(c.replace('_', ' ') for c in collections)}
Related pages to link:
{related_text}

Write the complete page now."""

    return _llm_chat(
        [{"role": "system", "content": _COMPARISON_SYNTH_SYSTEM}, {"role": "user", "content": user_prompt}],
        role="synthesize",
    )


def generate_comparison_page(
    topic: str,
    collections: list[str],
    force: bool = False,
    max_per_collection: int = _MAX_EVIDENCE_PER_COLLECTION,
) -> WikiPage | None:
    """Generate a comparison page across 2-4 collections on a topic.

    Pipeline: evidence selection -> per-source distillation (haiku) ->
    cross-source comparison (sonnet) -> page synthesis (sonnet).
    """
    collections = sorted(set(collections))
    if len(collections) < 2:
        print("  [wiki_gen] Comparison requires at least 2 unique collections")
        return None
    if len(collections) > _MAX_COMPARISON_COLLECTIONS:
        print(f"  [wiki_gen] Capping comparison to {_MAX_COMPARISON_COLLECTIONS} collections")
        collections = collections[:_MAX_COMPARISON_COLLECTIONS]
    topic_slug = _slug(topic)
    coll_slugs = "--".join(_slug(c.replace("_", " ")) for c in collections)
    page_id = f"comparison/{topic_slug}--{coll_slugs}"

    wm = get_wiki_manager()
    if not force and wm.page_exists(page_id):
        existing = wm.get_page(page_id)
        if existing and existing.status != "stale":
            print(f"  [wiki_gen] Comparison page already exists: {page_id}")
            return existing

    requested_collections = list(collections)
    print(f"  [wiki_gen] Generating comparison: '{topic}' across {collections}")

    # Stage 1: Evidence selection
    evidence_by_coll = select_evidence_for_comparison(
        topic, collections, max_per_collection=max_per_collection,
    )

    if len(evidence_by_coll) < 2:
        participating = list(evidence_by_coll.keys())
        print(f"  [wiki_gen] Only {len(participating)} collections have enough evidence, need 2+")
        return None

    collections = sorted(evidence_by_coll.keys())
    dropped_collections = sorted(set(requested_collections) - set(collections))
    if dropped_collections:
        print(f"  [wiki_gen] Dropped (insufficient evidence): {dropped_collections}")
    topic_slug = _slug(topic)
    coll_slugs = "--".join(_slug(c.replace("_", " ")) for c in collections)
    page_id = f"comparison/{topic_slug}--{coll_slugs}"

    all_evidence = [e for evs in evidence_by_coll.values() for e in evs]
    for coll, evs in evidence_by_coll.items():
        print(f"  [wiki_gen]   {coll}: {len(evs)} chunks")

    # Stage 2: Per-source distillation (haiku, one call per source)
    distilled: list[dict] = []
    for coll in collections:
        d = _distill_source_position(topic, coll, evidence_by_coll[coll])
        if d:
            distilled.append(d)
        else:
            print(f"  [wiki_gen]   Distillation failed for {coll}, skipping")

    if len(distilled) < 2:
        print(f"  [wiki_gen] Only {len(distilled)} sources distilled, need 2+")
        return None

    print(f"  [wiki_gen] Distilled {len(distilled)} source positions")

    # Stage 3: Cross-source comparison (sonnet)
    comparison = _compare_positions(topic, distilled)
    if not comparison:
        print(f"  [wiki_gen] Cross-source comparison failed")
        return None

    n_agreements = len(comparison.get("agreements", []))
    n_tensions = len(comparison.get("tensions", []))
    n_unique = len(comparison.get("unique_contributions", []))
    print(f"  [wiki_gen] Comparison: {n_agreements} agreements, {n_tensions} tensions, {n_unique} unique")

    # Stage 4: Page synthesis (sonnet)
    related = _find_related_pages(topic, "comparison", all_evidence)
    for coll in collections:
        source_page = f"source/{_slug(coll.replace('_', ' '))}"
        if source_page not in related and wm.page_exists(source_page):
            related.append(source_page)

    content = _synthesize_comparison(topic, collections, distilled, comparison, related)

    # Build claims from comparison analysis (validate chunk_ids against evidence)
    valid_ids = {e.chunk_id for e in all_evidence}
    claims: list[Claim] = []
    claim_idx = 0
    for a in comparison.get("agreements", []):
        cids = [cid for cid in a.get("chunk_ids", []) if cid in valid_ids]
        claims.append(Claim(
            claim_id=f"claim-{claim_idx:03d}",
            text=a.get("text", ""),
            chunk_ids=cids,
            collections=a.get("sources", []),
            support_count=len(cids),
            status="supported" if cids else "review",
            corroboration="high" if cids and len(a.get("sources", [])) >= 2 else "moderate" if cids else "low",
        ))
        claim_idx += 1
    for t in comparison.get("tensions", []):
        cids = [cid for cid in t.get("chunk_ids", []) if cid in valid_ids]
        claims.append(Claim(
            claim_id=f"claim-{claim_idx:03d}",
            text=t.get("text", ""),
            chunk_ids=cids,
            collections=list(t.get("source_positions", {}).keys()),
            support_count=len(cids),
            status="conflicted" if cids else "review",
            corroboration="mixed" if cids else "low",
        ))
        claim_idx += 1
    for u in comparison.get("unique_contributions", []):
        cids = [cid for cid in u.get("chunk_ids", []) if cid in valid_ids]
        claims.append(Claim(
            claim_id=f"claim-{claim_idx:03d}",
            text=u.get("text", ""),
            chunk_ids=cids,
            collections=[u.get("collection", "")],
            support_count=len(cids),
            status="supported" if cids else "review",
            corroboration="low",
        ))
        claim_idx += 1

    chunk_ids = sorted(set(e.chunk_id for e in all_evidence))
    coll_display = " vs ".join(c.replace("_", " ") for c in collections)
    title = f"{topic.replace('-', ' ').title()}: {coll_display}"

    page = WikiPage(
        page_id=page_id,
        page_type="comparison",
        title=title,
        slug=f"{topic_slug}--{coll_slugs}",
        version=(wm.get_page(page_id).version + 1) if wm.page_exists(page_id) else 1,
        source_collections=collections,
        source_chunk_count=len(chunk_ids),
        supporting_source_count=len(collections),
        corroboration_level=_page_corroboration(claims) if claims else "moderate",
        confidence=_page_confidence(claims) if claims else "medium",
        generation={
            "strategy": "comparison",
            "model_draft": _get_model("draft") or "default",
            "model_synthesize": _get_model("synthesize") or "default",
            "inputs_hash": _inputs_hash(chunk_ids),
            "source_versions": {},
            "claim_count": len(claims),
            "claims": [c.to_dict() for c in claims],
            "distilled_positions": distilled,
            "requested_collections": requested_collections,
            "dropped_collections": dropped_collections,
        },
        canonical_sources=[{"collection": c, "weight": "compared"} for c in collections],
        related_pages=related,
        content=content,
    )

    wm.save_page(page)
    print(f"  [wiki_gen] Saved comparison page: {page_id} ({len(claims)} claims)")
    return page
