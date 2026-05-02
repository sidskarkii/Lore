"""Wiki contradiction detection — find cross-page claim conflicts.

Deterministic candidate generation: groups claims by shared entity/concept
provenance, computes embedding similarity between pairs, flags high-similarity
pairs with negation asymmetry as potential contradictions. Stores results
in contradictions.json manifest.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ContradictionPair:
    claim_a_page: str
    claim_a_id: str
    claim_a_text: str
    claim_b_page: str
    claim_b_id: str
    claim_b_text: str
    similarity: float
    shared_subject: str
    confidence: str  # high, medium, low
    status: str = "unresolved"

    def to_dict(self) -> dict:
        return {
            "claim_a": {"page_id": self.claim_a_page, "claim_id": self.claim_a_id, "text": self.claim_a_text},
            "claim_b": {"page_id": self.claim_b_page, "claim_id": self.claim_b_id, "text": self.claim_b_text},
            "similarity": round(self.similarity, 4),
            "shared_subject": self.shared_subject,
            "confidence": self.confidence,
            "status": self.status,
        }


_NEGATION_PATTERNS = [
    re.compile(r'\bnot\b', re.I),
    re.compile(r'\bnever\b', re.I),
    re.compile(r'\bno\b', re.I),
    re.compile(r'\bwithout\b', re.I),
    re.compile(r'\boppos(?:e[ds]?|ite|ing)\b', re.I),
    re.compile(r'\breject(?:s|ed|ing)?\b', re.I),
    re.compile(r'\bdeni(?:es|ed|al)\b', re.I),
    re.compile(r'\bfail(?:s|ed|ure)?\b', re.I),
]


def _negation_count(text: str) -> int:
    return sum(1 for p in _NEGATION_PATTERNS if p.search(text))


def _negation_asymmetry(text_a: str, text_b: str) -> bool:
    return abs(_negation_count(text_a) - _negation_count(text_b)) >= 1


def _extract_subjects(claim: dict, page_id: str) -> set[str]:
    """Extract canonical entity names from claim provenance for grouping."""
    subjects = set()

    try:
        from .entities import get_entity_index
        from .cross_index import get_cross_index
        ei = get_entity_index()
        ci = get_cross_index()
        for chunk_id in claim.get("chunk_ids", []):
            cf = ci.by_chunk.get(chunk_id)
            if cf:
                for ent in cf.entities:
                    cluster = ei.resolve(ent)
                    if cluster:
                        subjects.add(cluster.canonical.lower())
    except Exception as e:
        print(f"  [contradictions] Subject extraction degraded: {e}")

    if not subjects and "/" in page_id:
        ptype, slug = page_id.split("/", 1)
        if ptype in ("entity", "concept"):
            subjects.add(slug.replace("-", " ").lower())

    return subjects


def detect_contradictions(
    min_similarity: float = 0.75,
    max_pairs: int = 50,
) -> list[ContradictionPair]:
    """Detect potential contradictions across all wiki page claims."""
    from .wiki import get_wiki_manager
    from .embed import embed_texts

    wm = get_wiki_manager()

    claims = []
    for meta in wm.list_pages():
        page = wm.get_page(meta["page_id"])
        if not page or not page.generation:
            continue
        for c in page.generation.get("claims", []):
            text = c.get("text", "")
            if not text:
                continue
            claims.append({
                "page_id": page.page_id,
                "claim_id": c.get("claim_id", ""),
                "text": text,
                "collections": set(c.get("collections", [])),
                "chunk_ids": c.get("chunk_ids", []),
                "subjects": _extract_subjects(c, page.page_id),
            })

    if len(claims) < 2:
        return []

    texts = [c["text"] for c in claims]
    embeddings = embed_texts(texts)
    vecs = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vecs = vecs / norms

    pairs: list[ContradictionPair] = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            if claims[i]["page_id"] == claims[j]["page_id"]:
                continue

            shared = claims[i]["subjects"] & claims[j]["subjects"]
            if not shared:
                continue

            sim = float(vecs[i] @ vecs[j])
            if sim < min_similarity:
                continue

            has_negation = _negation_asymmetry(claims[i]["text"], claims[j]["text"])
            if not has_negation:
                continue

            diff_source = not (claims[i]["collections"] & claims[j]["collections"])

            if sim >= 0.85 and diff_source:
                confidence = "high"
            elif sim >= 0.80 or diff_source:
                confidence = "medium"
            else:
                confidence = "low"

            pairs.append(ContradictionPair(
                claim_a_page=claims[i]["page_id"],
                claim_a_id=claims[i]["claim_id"],
                claim_a_text=claims[i]["text"],
                claim_b_page=claims[j]["page_id"],
                claim_b_id=claims[j]["claim_id"],
                claim_b_text=claims[j]["text"],
                similarity=sim,
                shared_subject=sorted(shared)[0],
                confidence=confidence,
            ))

    pairs.sort(key=lambda p: (-{"high": 3, "medium": 2, "low": 1}[p.confidence], -p.similarity))
    return pairs[:max_pairs]


def save_contradictions(pairs: list[ContradictionPair]):
    """Persist contradiction pairs to manifest (always overwrites)."""
    from .config import get_config
    cfg = get_config()
    path = cfg.data_dir / "wiki" / "manifests" / "contradictions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [p.to_dict() for p in pairs]
    path.write_text(json.dumps(data, indent=2))


def load_contradictions() -> list[dict]:
    """Load persisted contradiction pairs."""
    from .config import get_config
    cfg = get_config()
    path = cfg.data_dir / "wiki" / "manifests" / "contradictions.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
