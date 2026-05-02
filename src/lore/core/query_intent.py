"""Query intent detection — keyword-based heuristics for routing wiki vs chunks.

Classifies queries into intent signals that guide page-type boosting in
wiki_search and wiki-hint generation in search. Simple regex matching
covers ~80% of cases without LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class IntentSignals:
    is_comparison: bool = False
    is_conceptual: bool = False
    is_entity_lookup: bool = False
    is_exact_quote: bool = False
    is_citation: bool = False

    page_type_boosts: dict[str, float] = field(default_factory=dict)

    @property
    def wiki_favorable(self) -> bool:
        return self.is_comparison or self.is_conceptual or self.is_entity_lookup

    @property
    def chunk_favorable(self) -> bool:
        return self.is_exact_quote or self.is_citation

    @property
    def suggested_hint(self) -> str | None:
        if self.is_comparison:
            return "This looks like a comparison query — try wiki_search or wiki_generate_page(page_type='comparison') for synthesized cross-source analysis."
        if self.is_conceptual:
            return "This looks like a conceptual query — try wiki_search for synthesized concept/entity pages."
        if self.is_entity_lookup:
            return "This mentions specific entities — try wiki_search or wiki_get_page for synthesized entity pages."
        return None


_COMPARISON_PATTERNS = [
    re.compile(r'\b(?:compar(?:e|ing|ison|isons)?|differ(?:s|ent|ence|ences)?|contrast(?:s|ing)?|versus)\b', re.I),
    re.compile(r'\bvs\.?\b', re.I),
    re.compile(r'\b(?:similarities|differences)\s+between\b', re.I),
    re.compile(r'\bhow\s+(?:does|do|is|are)\s+\w+\s+(?:differ|compare)\b', re.I),
]

_CONCEPTUAL_PATTERNS = [
    re.compile(r'\b(?:what\s+is|what\s+are|explain|define|overview|summarize|summary\s+of)\b', re.I),
    re.compile(r'\b(?:concept|theory|principle|framework|meaning\s+of)\b', re.I),
    re.compile(r'\b(?:how\s+does|how\s+do)\b.*\bwork\b', re.I),
]

_EXACT_QUOTE_PATTERNS = [
    re.compile(r'"[^"]{10,}"'),
    re.compile(r'\b(?:exact\s+quote|exact\s+wording|verbatim)\b', re.I),
]

_CITATION_PATTERNS = [
    re.compile(r'\bpage\s+\d+\b', re.I),
    re.compile(r'\bchapter\s+\d+\b', re.I),
    re.compile(r'\b(?:paragraph|section)\s+\d+\b', re.I),
    re.compile(r'\b(?:timestamp|at\s+\d+:\d+)\b', re.I),
    re.compile(r'\b\d+:\d{2}(?::\d{2})?\b'),
]

# False positive guards — terms that contain comparison keywords but aren't comparisons
_COMPARISON_FALSE_POSITIVES = re.compile(
    r'\bvs\s*code\b', re.I,
)


def detect_query_intent(query: str) -> IntentSignals:
    signals = IntentSignals()

    is_comparison = any(p.search(query) for p in _COMPARISON_PATTERNS)
    if is_comparison and _COMPARISON_FALSE_POSITIVES.search(query):
        is_comparison = False

    if is_comparison:
        signals.is_comparison = True
        signals.page_type_boosts["comparison"] = 0.004
        signals.page_type_boosts["concept"] = 0.001

    if any(p.search(query) for p in _CONCEPTUAL_PATTERNS):
        signals.is_conceptual = True
        signals.page_type_boosts["concept"] = signals.page_type_boosts.get("concept", 0) + 0.003
        signals.page_type_boosts["entity"] = 0.002

    if any(p.search(query) for p in _EXACT_QUOTE_PATTERNS):
        signals.is_exact_quote = True

    if any(p.search(query) for p in _CITATION_PATTERNS):
        signals.is_citation = True

    if not signals.wiki_favorable and not signals.chunk_favorable:
        try:
            from .entities import get_entity_index
            idx = get_entity_index()
            words = query.split()
            for n in range(min(4, len(words)), 0, -1):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i+n])
                    if idx.resolve(phrase):
                        signals.is_entity_lookup = True
                        signals.page_type_boosts["entity"] = 0.002
                        break
                if signals.is_entity_lookup:
                    break
        except Exception:
            pass

    return signals
