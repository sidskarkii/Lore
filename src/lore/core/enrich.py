"""Chunk enrichment — programmatic (KeyBERT, spaCy) + multi-stage LLM pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

# ── Singletons (thread-safe) ────────────────────────────────────────

_enrichment_cache: dict[str, dict] | None = None
_cache_lock = threading.Lock()
_kw_model = None
_kw_lock = threading.Lock()
_nlp = None
_nlp_lock = threading.Lock()


def _get_enrichment_cache() -> dict[str, dict]:
    global _enrichment_cache
    with _cache_lock:
        if _enrichment_cache is None:
            cache_path = Path(__file__).resolve().parents[3] / ".enrichment_cache.json"
            if cache_path.exists():
                try:
                    _enrichment_cache = json.loads(cache_path.read_text())
                    print(f"  [enrich] Loaded {len(_enrichment_cache)} cached enrichments")
                except (json.JSONDecodeError, OSError) as e:
                    print(f"  [enrich] Cache file corrupt, starting fresh: {e}")
                    _enrichment_cache = {}
            else:
                _enrichment_cache = {}
    return _enrichment_cache


def _get_kw_model():
    global _kw_model
    with _kw_lock:
        if _kw_model is None:
            from keybert import KeyBERT
            print("  [enrich] Loading KeyBERT model...")
            _kw_model = KeyBERT(model="all-MiniLM-L6-v2")
            print("  [enrich] KeyBERT ready.")
    return _kw_model


def _get_nlp():
    global _nlp
    with _nlp_lock:
        if _nlp is None:
            import spacy
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                spacy.cli.download("en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm")
            print("  [enrich] spaCy ready.")
    return _nlp


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _save_enrichment_cache():
    """Persist the in-memory enrichment cache to disk."""
    if _enrichment_cache is None:
        return
    cache_path = Path(__file__).resolve().parents[3] / ".enrichment_cache.json"
    try:
        cache_path.write_text(json.dumps(_enrichment_cache, indent=2, default=str))
    except OSError as e:
        print(f"  [enrich] Failed to save cache: {e}")


# ── JSON extraction ─────────────────────────────────────────────────

def _extract_json(response: str) -> list[dict]:
    if not response:
        raise ValueError("Empty response")

    text = response.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate and (candidate[0] in "[{"):
                text = candidate
                break

    # Strip control chars
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', text)

    # Fix trailing commas before ] or } (common LLM mistake)
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # Fix single-quoted JSON keys/values → double-quoted
    # Only replace quotes that look like JSON delimiters, not apostrophes in text
    if "'" in text:
        text = re.sub(r"(?<=[\[{,:])\s*'|'\s*(?=[:,\]}])", '"', text)

    try:
        results = json.loads(text)
    except json.JSONDecodeError:
        candidates = []
        for pattern in [r'\[.*\]', r'\{.*\}']:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(re.sub(r',\s*([}\]])', r'\1', match.group()))
                    candidates.append((len(match.group()), parsed))
                except json.JSONDecodeError:
                    pass

        if candidates:
            candidates.sort(key=lambda x: -x[0])
            results = candidates[0][1]
        else:
            raise

    if not isinstance(results, list):
        results = [results]
    return results


# ── Programmatic enrichment (no LLM) ────────────────────────────────

def enrich_programmatic(chunks: list[dict]) -> list[dict]:
    texts = [c["text"] for c in chunks]
    keywords_list = _extract_keywords_batch(texts)
    entities_list = _extract_entities_batch(texts)
    for chunk, kws, ents in zip(chunks, keywords_list, entities_list):
        chunk["keywords"] = ", ".join(kws)
        chunk["entities"] = json.dumps(ents) if ents else ""
    return chunks


def _extract_keywords_batch(texts: list[str]) -> list[list[str]]:
    try:
        kw_model = _get_kw_model()
        results = []
        for text in texts:
            if len(text.split()) < 10:
                results.append([])
                continue
            kws = kw_model.extract_keywords(
                text, keyphrase_ngram_range=(1, 2),
                stop_words="english", top_n=6, use_mmr=True, diversity=0.5,
            )
            results.append([kw for kw, _ in kws])
        return results
    except ImportError:
        print("  KeyBERT not installed — skipping keyword extraction")
        return [[] for _ in texts]


def _extract_entities_batch(texts: list[str]) -> list[list[dict]]:
    try:
        nlp = _get_nlp()
        results = []
        for doc in nlp.pipe(texts, batch_size=32):
            ents = []
            seen: set[tuple[str, str]] = set()
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART", "EVENT", "LAW"):
                    key = (ent.text, ent.label_)
                    if key not in seen:
                        seen.add(key)
                        ents.append({"name": ent.text, "type": ent.label_})
            results.append(ents)
        return results
    except ImportError:
        print("  spaCy not installed — skipping entity extraction")
        return [[] for _ in texts]


# ── Prompts ──────────────────────────────────────────────────────────

_STAGE2_SYSTEM = """You are enriching document chunks for a retrieval system. Extract stable metadata faithful to the chunk text.

Rules:
- The chunk text is the source of truth. Candidate cues and rolling keys are hints, not facts.
- Reuse an existing concept key when the chunk clearly matches it. Create new keys only for genuinely new concepts.
- Prefer canonical, reusable tags over synonyms (e.g. always "deception-as-advantage" not sometimes "strategic-misdirection").
- Output strict JSON only. No prose. No markdown fences.

Schema (return one object per passage in a JSON array):
{{
  "title": "string, 4-10 words, specific and concrete",
  "summary": "string, 1-2 sentences grounded in this chunk only",
  "tags": ["broad topical tags, lowercase-hyphens, 3-6"],
  "concept_tags": ["canonical concept keys from rolling dict or new, 2-5"],
  "related_tags": ["subset of rolling keys this chunk references, 0-4"],
  "importance": 1,
  "why_important": "one sentence justification",
  "questions": ["2-3 specific questions this chunk can answer"],
  "self_contained": true,
  "confidence": "fact|empirical|opinion|anecdote|hypothesis",
  "semantic_key": "2-5 word subtopic identifier"
}}

Importance rubric:
1 = minor detail or filler
2 = supporting context or background
3 = useful core point
4 = important concept, mechanism, or transition
5 = critical claim, definition, or turning point

Confidence guide:
- fact: presented as established truth ("X is Y")
- empirical: backed by data, study, or experiment ("studies show...")
- opinion: author's view ("I believe...", "in my experience...")
- anecdote: story or example illustrating a point
- hypothesis: speculation or possibility ("it's possible that...")

Few-shot example:
Rolling keys: retrieval-augmented-generation (8x), vector-indexing (5x), hallucination-mitigation (3x)
Chunk: "The system embeds passages into a vector index, then retrieves them at query time to reduce unsupported model claims."
Output:
{{
  "title": "Retrieval Reduces Unsupported Claims",
  "summary": "Passages are embedded into a vector index and retrieved during answering to reduce unsupported model outputs.",
  "tags": ["rag", "retrieval", "vector-search"],
  "concept_tags": ["retrieval-augmented-generation", "vector-indexing", "hallucination-mitigation"],
  "related_tags": ["vector-indexing", "hallucination-mitigation"],
  "importance": 4,
  "why_important": "Describes the core mechanism and purpose of the retrieval system.",
  "questions": ["How does retrieval reduce hallucination?", "What role does the vector index play in RAG?"],
  "self_contained": true,
  "confidence": "fact",
  "semantic_key": "rag retrieval mechanism"
}}"""

_STAGE2_USER = """Document: {book_title}
Section: {section_heading}
Previous chunk: {prev_context}

Rolling concept keys:
{rolling_keys}

Candidate cues from prior analysis:
Keywords: {keywords}
Entities: {entities}

Passages:
{chunks}

Return a JSON array with one object per passage, matching the schema exactly."""

_STAGE3_SYSTEM = """You are updating a section-level synthesis for a retrieval system.

Rules:
- Original chunk text is primary evidence. Chunk metadata is structured hints.
- Preserve stable concepts; avoid synonym drift. Prefer refining existing ledger keys over inventing new ones.
- Update the running summary only with claims supported by this batch.
- Output strict JSON only. No prose outside JSON.

Schema:
{{
  "section_summary": "string, 2-5 sentences reflecting cumulative section state",
  "section_themes": ["stable section-level concepts, 3-8"],
  "key_entities": ["people, organizations, works mentioned"],
  "ledger_updates": [
    {{"concept_tag": "string", "action": "keep|refine|add|downweight", "evidence": "short grounded note"}}
  ],
  "notable_points": ["important developments or claims, 2-6"],
  "open_questions_or_tensions": ["contrasts, ambiguities, unresolved points, 0-3"]
}}"""

_STAGE3_USER = """Document: {book_title}
Section: {section_name}

Current running summary:
{running_summary}

Current concept ledger:
{concept_ledger}

Chunk batch (with Stage 2 metadata):
{chunks}

Update the section summary to reflect the cumulative state after this batch.
Update the concept ledger with only justified changes.
Note contradictions or shifts in emphasis if present.
Return JSON matching the schema exactly."""

_BOOK_SUMMARY_PROMPT = """You are creating an overview of "{book_title}" by {author}.

Below are section summaries with their concept tags and themes.

Return a JSON object with:
- "overview": 4-6 sentence overview
- "main_themes": Array of 3-6 themes, each with "theme" and "description" fields
- "key_takeaways": Array of 5-10 key insights or arguments
- "tags": Array of 8-12 book-level tags (lowercase, hyphens)
- "cross_section_patterns": Array of 2-5 concepts that recur across multiple sections

Return ONLY valid JSON, no other text.

Table of contents:
{toc}

Top recurring concepts across sections:
{concept_summary}

Section summaries:
{section_summaries}"""


# ── LLM call infrastructure ─────────────────────────────────────────

_FALLBACK_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-3-12b-it:free",
]


class _ModerationBlock(Exception):
    pass


class _RateLimit(Exception):
    pass


class _Timeout(Exception):
    pass


def _classify_error(e: Exception) -> Exception:
    err_str = str(e)
    err_type = type(e).__name__.lower()
    if "403" in err_str or "moderation" in err_str.lower() or "flagged" in err_str.lower():
        return _ModerationBlock(err_str)
    if "429" in err_str or "rate" in err_str.lower():
        return _RateLimit(err_str)
    if "timeout" in err_str.lower() or "timed out" in err_str.lower() or "timeout" in err_type:
        return _Timeout(err_str)
    return e


def _llm_call(provider, messages, model=None) -> str:
    try:
        response = provider.chat(messages, model=model)
        if response is None:
            raise ValueError("Provider returned None")
        return response
    except (_ModerationBlock, _RateLimit, _Timeout):
        raise
    except Exception as e:
        raise _classify_error(e) from e


def _llm_call_with_retry(provider, messages, max_retries: int = 2) -> str:
    """Retry on rate limits and timeouts. Raises _ModerationBlock immediately."""
    for attempt in range(max_retries + 1):
        try:
            return _llm_call(provider, messages)
        except _ModerationBlock:
            raise
        except (_RateLimit, _Timeout):
            if attempt < max_retries:
                wait = 2 ** attempt * 3
                print(f"  [enrich] Retryable error, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            raise
        except Exception:
            if attempt < max_retries:
                continue
            raise


def _llm_call_with_fallback(provider, messages) -> str:
    """Try fallback models. Used in end-of-stage retry pass."""
    errors = []
    for fallback in _FALLBACK_MODELS:
        try:
            response = _llm_call(provider, messages, model=fallback)
            print(f"  [enrich] Fallback succeeded: {fallback}")
            return response
        except _RateLimit:
            errors.append(f"{fallback}: rate limited")
            time.sleep(8)
            continue
        except Exception as e:
            errors.append(f"{fallback}: {e}")
            continue
    raise RuntimeError(f"All fallback models failed: {'; '.join(errors)}")


# ── Chunk field helpers ──────────────────────────────────────────────

_CACHE_FIELDS = ("title", "summary", "keywords", "concept_tags", "importance", "questions",
                 "semantic_key", "self_contained", "confidence", "why_important")


def _apply_enrichment(chunk: dict, enrichment: dict):
    """Apply enrichment data to a chunk. All-or-nothing — doesn't partial write. Also caches."""
    update = {
        "title": enrichment.get("title", ""),
        "summary": enrichment.get("summary", ""),
        "keywords": ", ".join(enrichment.get("tags", [])),
        "concept_tags": ", ".join(enrichment.get("concept_tags", [])),
        "importance": int(enrichment.get("importance", 3)),
        "semantic_key": enrichment.get("semantic_key", ""),
        "questions": ", ".join(enrichment.get("questions", [])),
        "self_contained": enrichment.get("self_contained", True),
        "confidence": enrichment.get("confidence", ""),
        "why_important": enrichment.get("why_important", ""),
    }
    chunk.update(update)

    cache = _get_enrichment_cache()
    h = _content_hash(chunk.get("text", ""))
    if h:
        cache[h] = {k: v for k, v in update.items()}


# ── Rolling key dictionary ──────────────────────────────────────────

class _RollingKeyDict:
    """Accumulates concept tags across chunks, MDKeyChunker-style."""

    def __init__(self, max_keys: int = 100):
        self.max_keys = max_keys
        self._keys: dict[str, dict] = {}

    def update_from_chunk(self, concept_tags: list[str], chunk_idx: int):
        for tag in concept_tags:
            tag = tag.strip().lower()
            if not tag:
                continue
            if tag in self._keys:
                self._keys[tag]["last_chunk"] = chunk_idx
                self._keys[tag]["count"] += 1
            else:
                self._keys[tag] = {"first_chunk": chunk_idx, "last_chunk": chunk_idx, "count": 1}
        self._prune()

    def _prune(self):
        if len(self._keys) <= self.max_keys:
            return
        sorted_keys = sorted(self._keys.items(), key=lambda x: x[1]["last_chunk"], reverse=True)
        self._keys = dict(sorted_keys[:self.max_keys])

    def format_for_prompt(self) -> str:
        if not self._keys:
            return "(none yet — first chunk)"
        parts = []
        for key, info in sorted(self._keys.items(), key=lambda x: -x[1]["count"]):
            parts.append(f"- {key} ({info['count']}x)")
        return "\n".join(parts)


def _apply_cache(chunk: dict, cached: dict):
    """Apply cached enrichment including concept_tags and importance."""
    for field in _CACHE_FIELDS:
        if field in cached and cached[field] is not None:
            chunk[field] = cached[field]
    if "importance" not in chunk or chunk.get("importance") is None:
        chunk["importance"] = 3


# ── Stage 2: Chunk-level enrichment ─────────────────────────────────

_MAX_TOKENS_PER_PASS = 5000


def enrich_chunks_stage2(
    chunks: list[dict],
    provider,
    book_title: str = "",
    calls_per_min: float = 7.5,
    on_progress=None,
) -> list[dict]:
    """Stage 2: Chunk-level enrichment with rolling key dictionary and Stage 1 cues."""
    min_interval = 60.0 / calls_per_min
    cache = _get_enrichment_cache()
    batch_size = 5

    enrichable = []
    cached_count = 0
    for i, c in enumerate(chunks):
        if len(c["text"].split()) < 15:
            continue
        h = _content_hash(c["text"])
        if h in cache and all(cache[h].get(f) for f in ("title", "summary", "questions")):
            _apply_cache(c, cache[h])
            cached_count += 1
        else:
            enrichable.append((i, c))

    total_batches = (len(enrichable) + batch_size - 1) // batch_size
    print(f"  [stage2] Chunk enrichment: {cached_count} cached, {len(enrichable)} need LLM ({total_batches} batches)")

    max_keys = min(max(len(chunks) // 3, 50), 200)
    rolling_keys = _RollingKeyDict(max_keys=max_keys)

    for c in chunks[:enrichable[0][0]] if enrichable else []:
        tags = [t.strip() for t in c.get("concept_tags", "").split(",") if t.strip()]
        rolling_keys.update_from_chunk(tags, 0)

    prev_title = ""
    prev_summary = ""
    prev_tags = ""
    last_call = 0.0
    failed: list[tuple[list[tuple[int, dict]], str]] = []

    system_msg = {"role": "system", "content": _STAGE2_SYSTEM}

    for batch_start in range(0, len(enrichable), batch_size):
        batch = enrichable[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1

        elapsed = time.time() - last_call
        if last_call > 0 and elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        if on_progress:
            on_progress(batch_num, total_batches, cached_count)

        section = batch[0][1].get("section_heading", "") if batch else ""

        chunks_text = ""
        for idx, (_, chunk) in enumerate(batch):
            kw = chunk.get("keywords", "")
            ents = chunk.get("entities", "")
            cue_line = f"\n[Cues: keywords={kw}, entities={ents}]" if (kw or ents) else ""
            chunks_text += f"\n--- Passage {idx + 1} ---{cue_line}\n{chunk['text']}\n"

        prev_ctx = f"Title: {prev_title}\nSummary: {prev_summary}\nConcept tags: {prev_tags}" if prev_title else "(first batch)"

        user_msg = {"role": "user", "content": _STAGE2_USER.format(
            book_title=book_title or "Unknown",
            section_heading=section or "(untitled)",
            prev_context=prev_ctx,
            rolling_keys=rolling_keys.format_for_prompt(),
            keywords="(per-passage above)",
            entities="(per-passage above)",
            chunks=chunks_text,
        )}

        try:
            last_call = time.time()
            response = _llm_call_with_retry(provider, [system_msg, user_msg])
            results = _extract_json(response)

            if len(results) != len(batch):
                print(f"  [stage2] Batch {batch_num}: expected {len(batch)} results, got {len(results)} — marking failed")
                failed.append((batch, section))
                continue

            for (orig_idx, chunk), enrichment in zip(batch, results):
                _apply_enrichment(chunk, enrichment)
                tags = enrichment.get("concept_tags", [])
                rolling_keys.update_from_chunk(tags, orig_idx)

            last_enriched = results[-1]
            prev_title = last_enriched.get("title", "")
            prev_summary = last_enriched.get("summary", "")
            prev_tags = ", ".join(last_enriched.get("concept_tags", []))
        except Exception as e:
            print(f"  [stage2] Batch {batch_num} failed: {type(e).__name__}")
            failed.append((batch, section))

    if failed:
        print(f"  [stage2] Retrying {len(failed)} failed batches with fallback models...")
        for batch, section in failed:
            time.sleep(min_interval)
            chunks_text = "\n".join(f"--- Passage {i+1} ---\n{c['text']}" for i, (_, c) in enumerate(batch))
            kw = batch[0][1].get("keywords", "") if batch else ""
            ents = batch[0][1].get("entities", "") if batch else ""
            user_msg = {"role": "user", "content": _STAGE2_USER.format(
                book_title=book_title or "Unknown",
                section_heading=section or "(untitled)",
                prev_context="(retry — no prior context)",
                rolling_keys=rolling_keys.format_for_prompt(),
                keywords=kw, entities=ents, chunks=chunks_text,
            )}
            try:
                response = _llm_call_with_fallback(provider, [system_msg, user_msg])
                results = _extract_json(response)
                if len(results) != len(batch):
                    print(f"  [stage2] Fallback returned {len(results)} results for {len(batch)} chunks — skipping")
                else:
                    for (orig_idx, chunk), enrichment in zip(batch, results):
                        _apply_enrichment(chunk, enrichment)
                        rolling_keys.update_from_chunk(enrichment.get("concept_tags", []), orig_idx)
            except Exception as e:
                print(f"  [stage2] Retry permanently failed: {e}")

    _save_enrichment_cache()
    return chunks


# ── Stage 3: Section/chapter summaries ──────────────────────────────

def _format_chunk_batch_for_stage3(batch: list[dict]) -> str:
    """Format chunks with Stage 2 metadata for Stage 3 prompt."""
    items = []
    for c in batch:
        items.append(json.dumps({
            "chunk_title": c.get("title", ""),
            "chunk_summary": c.get("summary", ""),
            "concept_tags": c.get("concept_tags", ""),
            "importance": c.get("importance", 3),
            "why_important": c.get("why_important", ""),
            "chunk_text": c.get("text", ""),
        }))
    return "\n".join(items)


def _format_concept_ledger(ledger: dict[str, int]) -> str:
    if not ledger:
        return "(empty — first pass)"
    return "\n".join(f"- {tag} ({count}x)" for tag, count in sorted(ledger.items(), key=lambda x: -x[1]))


def enrich_section_stage3(
    section_chunks: list[dict],
    provider,
    book_title: str = "",
    section_name: str = "",
    calls_per_min: float = 7.5,
    on_progress=None,
) -> dict:
    """Stage 3: Section summary with concept ledger and Stage 2 metadata."""
    min_interval = 60.0 / calls_per_min

    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_tokens = 0
    for c in section_chunks:
        ct = len(c.get("text", "")) // 4
        if current_tokens + ct > _MAX_TOKENS_PER_PASS and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(c)
        current_tokens += ct
    if current_batch:
        batches.append(current_batch)

    total_passes = len(batches)
    print(f"  [stage3] Section '{section_name}': {total_passes} passes over {len(section_chunks)} chunks")

    running_summary = "No summary yet — this is the first pass."
    concept_ledger: dict[str, int] = {}
    all_key_entities: list[str] = []
    all_notable_points: list[str] = []
    all_tensions: list[str] = []
    failed_passes: list[tuple[int, list[dict], str, dict]] = []
    last_call = 0.0

    system_msg = {"role": "system", "content": _STAGE3_SYSTEM}

    for i, batch in enumerate(batches):
        if on_progress:
            on_progress(section_name, i + 1, total_passes)

        if i > 0:
            elapsed = time.time() - last_call
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

        for c in batch:
            for tag in c.get("concept_tags", "").split(","):
                tag = tag.strip()
                if tag:
                    concept_ledger[tag] = concept_ledger.get(tag, 0) + 1

        user_msg = {"role": "user", "content": _STAGE3_USER.format(
            book_title=book_title or "Unknown",
            section_name=section_name,
            running_summary=running_summary,
            concept_ledger=_format_concept_ledger(concept_ledger),
            chunks=_format_chunk_batch_for_stage3(batch),
        )}

        summary_before = running_summary
        ledger_before = dict(concept_ledger)

        try:
            last_call = time.time()
            response = _llm_call_with_retry(provider, [system_msg, user_msg])
            result = _extract_json(response)
            if isinstance(result, list):
                result = result[0]

            if result.get("section_summary"):
                running_summary = result["section_summary"]
            for entity in result.get("key_entities", []):
                if entity not in all_key_entities:
                    all_key_entities.append(entity)
            for point in result.get("notable_points", []):
                if point not in all_notable_points:
                    all_notable_points.append(point)
            for tension in result.get("open_questions_or_tensions", []):
                if tension not in all_tensions:
                    all_tensions.append(tension)
            for theme in result.get("section_themes", []):
                if theme and theme not in concept_ledger:
                    concept_ledger[theme] = 1
            for update in result.get("ledger_updates", []):
                tag = update.get("concept_tag", "")
                action = update.get("action", "keep")
                if tag and action in ("add", "refine"):
                    concept_ledger[tag] = concept_ledger.get(tag, 0) + 1
                elif tag and action == "downweight" and tag in concept_ledger:
                    concept_ledger[tag] = max(0, concept_ledger[tag] - 1)
        except Exception as e:
            print(f"  [stage3] Pass {i+1}/{total_passes} failed: {type(e).__name__}")
            failed_passes.append((i, batch, summary_before, ledger_before))

    if failed_passes:
        print(f"  [stage3] Retrying {len(failed_passes)} failed passes with fallback models...")
        for pass_idx, batch, summary_at_failure, ledger_at_failure in failed_passes:
            time.sleep(min_interval)
            user_msg = {"role": "user", "content": _STAGE3_USER.format(
                book_title=book_title or "Unknown",
                section_name=section_name,
                running_summary=summary_at_failure,
                concept_ledger=_format_concept_ledger(ledger_at_failure),
                chunks=_format_chunk_batch_for_stage3(batch),
            )}
            try:
                response = _llm_call_with_fallback(provider, [system_msg, user_msg])
                result = _extract_json(response)
                if isinstance(result, list):
                    result = result[0]
                if result.get("section_summary"):
                    running_summary = result["section_summary"]
                for theme in result.get("section_themes", []):
                    if theme and theme not in concept_ledger:
                        concept_ledger[theme] = 1
                for update in result.get("ledger_updates", []):
                    tag = update.get("concept_tag", "")
                    action = update.get("action", "keep")
                    if tag and action in ("add", "refine"):
                        concept_ledger[tag] = concept_ledger.get(tag, 0) + 1
                for entity in result.get("key_entities", []):
                    if entity not in all_key_entities:
                        all_key_entities.append(entity)
                for point in result.get("notable_points", []):
                    if point not in all_notable_points:
                        all_notable_points.append(point)
                for tension in result.get("open_questions_or_tensions", []):
                    if tension not in all_tensions:
                        all_tensions.append(tension)
            except Exception as e:
                print(f"  [stage3] Retry pass {pass_idx+1} permanently failed: {e}")

    return {
        "summary": running_summary,
        "key_concepts": list(concept_ledger.keys()),
        "key_entities": all_key_entities,
        "notable_points": all_notable_points,
        "tensions": all_tensions,
        "concept_ledger": concept_ledger,
    }


# ── Stage 4: Book-level summary ─────────────────────────────────────

def enrich_book_stage4(
    section_summaries: list[dict],
    provider,
    book_title: str = "",
    author: str = "",
    toc: list[str] | None = None,
) -> dict:
    from collections import Counter

    toc_text = "\n".join(f"- {t}" for t in (toc or []))

    summaries_parts = []
    concept_sections: dict[str, int] = {}
    for s in section_summaries:
        section_line = f"### {s.get('section', 'Unknown')}\n{s.get('summary', '')}"
        concepts = s.get("key_concepts", [])
        if concepts:
            section_line += f"\nConcepts: {', '.join(concepts)}"
        tensions = s.get("tensions", [])
        if tensions:
            section_line += f"\nTensions: {', '.join(tensions)}"
        summaries_parts.append(section_line)
        seen_in_section = set()
        ledger = s.get("concept_ledger", {})
        for tag in (ledger.keys() if ledger else concepts):
            if tag not in seen_in_section:
                seen_in_section.add(tag)
                concept_sections[tag] = concept_sections.get(tag, 0) + 1

    concept_summary = "\n".join(
        f"- {tag} (across {count} sections)"
        for tag, count in sorted(concept_sections.items(), key=lambda x: -x[1])[:20]
    ) if concept_sections else "(no concept data)"

    prompt = _BOOK_SUMMARY_PROMPT.format(
        book_title=book_title or "Unknown",
        author=author or "Unknown",
        toc=toc_text or "(no TOC available)",
        concept_summary=concept_summary,
        section_summaries="\n\n".join(summaries_parts),
    )
    try:
        response = _llm_call_with_retry(provider, [{"role": "user", "content": prompt}])
        result = _extract_json(response)
        if isinstance(result, list):
            result = result[0]
        return result
    except Exception as e:
        print(f"  [stage4] Book summary failed: {type(e).__name__}: {e}")
        return {"overview": "", "main_themes": [], "key_takeaways": [], "tags": [],
                "cross_section_patterns": [], "_error": str(e)}
