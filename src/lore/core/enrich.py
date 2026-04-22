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
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', text)

    try:
        results = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            results = json.loads(match.group())
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

_CHUNK_ENRICH_PROMPT = """You are analyzing passages from "{book_title}"{section_context}.

For each passage, return a JSON array with one object per passage containing:
- "title": Concise title (3-8 words) capturing the main point
- "summary": 1-2 sentence summary of what this passage says
- "tags": Array of 3-6 tags (lowercase, hyphens, no spaces) for topic categorization
- "concept_tags": Array of 2-5 specific concept tags that describe the IDEAS in this passage. These must be specific enough to connect similar ideas across different books. NOT generic topics — specific concepts. Examples: "deception-as-advantage", "reciprocity-principle", "information-asymmetry", "leader-discipline-relationship", "ego-driven-failure". Think: if another book discusses the same concept, would this tag match?
- "importance": Integer 1-5 rating how central this passage is to the chapter's argument (5=core thesis, 1=tangential)
- "semantic_key": 2-5 word subtopic identifier

Return ONLY a valid JSON array, no other text.

Passages:
{chunks}"""

_SECTION_PROGRESSIVE_PROMPT = """You are reading "{book_title}", section "{section_name}".

Here is your running summary so far:
{running_summary}

Now read the next passages and update your summary.

Return a JSON object with:
- "running_summary": Updated 3-5 sentence summary incorporating the new material
- "key_concepts": Updated array of main concepts (add new ones, keep existing)
- "key_entities": Array of people, organizations, or works mentioned so far
- "chunk_titles": Array of objects {{"title": "...", "importance": 1-5}} one per passage below

Return ONLY valid JSON, no other text.

Next passages:
{chunks}"""

_BOOK_SUMMARY_PROMPT = """You are creating an overview of "{book_title}" by {author}.

Below are the section summaries and table of contents.

Return a JSON object with:
- "overview": 4-6 sentence book overview
- "main_themes": Array of 3-6 main themes with brief descriptions
- "key_takeaways": Array of 5-10 key insights or arguments
- "tags": Array of 8-12 book-level tags (lowercase, hyphens)

Return ONLY valid JSON, no other text.

Table of contents:
{toc}

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

_CHUNK_FIELDS = ("title", "summary", "keywords", "concept_tags", "importance", "semantic_key")
_CACHE_FIELDS = ("title", "summary", "keywords", "concept_tags", "importance", "questions", "semantic_key")


def _apply_enrichment(chunk: dict, enrichment: dict):
    """Apply enrichment data to a chunk. All-or-nothing — doesn't partial write. Also caches."""
    update = {
        "title": enrichment.get("title", ""),
        "summary": enrichment.get("summary", ""),
        "keywords": ", ".join(enrichment.get("tags", [])),
        "concept_tags": ", ".join(enrichment.get("concept_tags", [])),
        "importance": int(enrichment.get("importance", 3)),
        "semantic_key": enrichment.get("semantic_key", ""),
    }
    chunk.update(update)

    cache = _get_enrichment_cache()
    h = _content_hash(chunk.get("text", ""))
    if h:
        cache[h] = {k: v for k, v in update.items()}


def _apply_cache(chunk: dict, cached: dict):
    """Apply cached enrichment including concept_tags and importance."""
    for field in _CACHE_FIELDS:
        if cached.get(field):
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
    """Stage 2: Chunk-level titles, tags, concept_tags, importance. Context-aware."""
    min_interval = 60.0 / calls_per_min
    cache = _get_enrichment_cache()
    batch_size = 5

    enrichable = []
    cached_count = 0
    for i, c in enumerate(chunks):
        if len(c["text"].split()) < 15:
            continue
        h = _content_hash(c["text"])
        if h in cache:
            _apply_cache(c, cache[h])
            cached_count += 1
        else:
            enrichable.append((i, c))

    total_batches = (len(enrichable) + batch_size - 1) // batch_size
    print(f"  [stage2] Chunk enrichment: {cached_count} cached, {len(enrichable)} need LLM ({total_batches} batches)")

    last_call = 0.0
    failed: list[tuple[list[tuple[int, dict]], str]] = []

    for batch_start in range(0, len(enrichable), batch_size):
        batch = enrichable[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1

        elapsed = time.time() - last_call
        if last_call > 0 and elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        if on_progress:
            on_progress(batch_num, total_batches, cached_count)

        section = batch[0][1].get("section_heading", "") if batch else ""
        section_ctx = f", section \"{section}\"" if section else ""

        chunks_text = ""
        for idx, (_, chunk) in enumerate(batch):
            chunks_text += f"\n--- Passage {idx + 1} ---\n{chunk['text']}\n"

        prompt = _CHUNK_ENRICH_PROMPT.format(
            book_title=book_title or "Unknown",
            section_context=section_ctx,
            chunks=chunks_text,
        )

        try:
            last_call = time.time()
            response = _llm_call_with_retry(provider, [{"role": "user", "content": prompt}])
            results = _extract_json(response)

            if len(results) != len(batch):
                print(f"  [stage2] Batch {batch_num}: expected {len(batch)} results, got {len(results)} — marking failed")
                failed.append((batch, section_ctx))
                continue

            for (orig_idx, chunk), enrichment in zip(batch, results):
                _apply_enrichment(chunk, enrichment)
        except Exception as e:
            print(f"  [stage2] Batch {batch_num} failed: {type(e).__name__}")
            failed.append((batch, section_ctx))

    if failed:
        print(f"  [stage2] Retrying {len(failed)} failed batches with fallback models...")
        for batch, section_ctx in failed:
            time.sleep(min_interval)
            chunks_text = "\n".join(f"--- Passage {i+1} ---\n{c['text']}" for i, (_, c) in enumerate(batch))
            prompt = _CHUNK_ENRICH_PROMPT.format(
                book_title=book_title or "Unknown",
                section_context=section_ctx,
                chunks=chunks_text,
            )
            try:
                response = _llm_call_with_fallback(provider, [{"role": "user", "content": prompt}])
                results = _extract_json(response)
                if len(results) != len(batch):
                    print(f"  [stage2] Fallback returned {len(results)} results for {len(batch)} chunks — skipping")
                else:
                    for (orig_idx, chunk), enrichment in zip(batch, results):
                        _apply_enrichment(chunk, enrichment)
            except Exception as e:
                print(f"  [stage2] Retry permanently failed: {e}")

    _save_enrichment_cache()
    return chunks


# ── Stage 3: Section/chapter summaries ──────────────────────────────

def enrich_section_stage3(
    section_chunks: list[dict],
    provider,
    book_title: str = "",
    section_name: str = "",
    calls_per_min: float = 7.5,
    on_progress=None,
) -> dict:
    """Stage 3: Section summary via progressive passes over original text.

    Always uses progressive passes. Failed passes are retried at end
    with fallback models using the running_summary state from just
    before the failure.
    """
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
    all_key_concepts: list[str] = []
    all_key_entities: list[str] = []
    chunk_titles: list[dict] = []
    failed_passes: list[tuple[int, list[dict], str]] = []
    last_call = 0.0

    for i, batch in enumerate(batches):
        if on_progress:
            on_progress(section_name, i + 1, total_passes)

        if i > 0:
            elapsed = time.time() - last_call
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

        chunks_text = "\n\n".join(bc["text"] for bc in batch)
        prompt = _SECTION_PROGRESSIVE_PROMPT.format(
            book_title=book_title or "Unknown",
            section_name=section_name,
            running_summary=running_summary,
            chunks=chunks_text,
        )

        summary_before_pass = running_summary

        try:
            last_call = time.time()
            response = _llm_call_with_retry(provider, [{"role": "user", "content": prompt}])
            result = _extract_json(response)
            if isinstance(result, list):
                result = result[0]
            running_summary = result.get("running_summary", running_summary)
            chunk_titles.extend(result.get("chunk_titles", []))
            for concept in result.get("key_concepts", []):
                if concept not in all_key_concepts:
                    all_key_concepts.append(concept)
            for entity in result.get("key_entities", []):
                if entity not in all_key_entities:
                    all_key_entities.append(entity)
        except Exception as e:
            print(f"  [stage3] Pass {i+1}/{total_passes} failed: {type(e).__name__}")
            failed_passes.append((i, batch, summary_before_pass))

    if failed_passes:
        print(f"  [stage3] Retrying {len(failed_passes)} failed passes with fallback models...")
        for pass_idx, batch, summary_at_failure in failed_passes:
            time.sleep(min_interval)
            chunks_text = "\n\n".join(bc["text"] for bc in batch)
            prompt = _SECTION_PROGRESSIVE_PROMPT.format(
                book_title=book_title or "Unknown",
                section_name=section_name,
                running_summary=summary_at_failure,
                chunks=chunks_text,
            )
            try:
                response = _llm_call_with_fallback(provider, [{"role": "user", "content": prompt}])
                result = _extract_json(response)
                if isinstance(result, list):
                    result = result[0]
                for concept in result.get("key_concepts", []):
                    if concept not in all_key_concepts:
                        all_key_concepts.append(concept)
                for entity in result.get("key_entities", []):
                    if entity not in all_key_entities:
                        all_key_entities.append(entity)
            except Exception as e:
                print(f"  [stage3] Retry pass {pass_idx+1} permanently failed: {e}")

    return {
        "summary": running_summary,
        "key_concepts": all_key_concepts,
        "key_entities": all_key_entities,
        "chunk_titles": chunk_titles,
    }


# ── Stage 4: Book-level summary ─────────────────────────────────────

def enrich_book_stage4(
    section_summaries: list[dict],
    provider,
    book_title: str = "",
    author: str = "",
    toc: list[str] | None = None,
) -> dict:
    toc_text = "\n".join(f"- {t}" for t in (toc or []))
    summaries_text = "\n\n".join(
        f"### {s.get('section', 'Unknown')}\n{s.get('summary', '')}"
        for s in section_summaries
    )
    prompt = _BOOK_SUMMARY_PROMPT.format(
        book_title=book_title or "Unknown",
        author=author or "Unknown",
        toc=toc_text or "(no TOC available)",
        section_summaries=summaries_text,
    )
    try:
        response = _llm_call_with_retry(provider, [{"role": "user", "content": prompt}])
        result = _extract_json(response)
        if isinstance(result, list):
            result = result[0]
        return result
    except Exception as e:
        print(f"  [stage4] Book summary failed: {type(e).__name__}: {e}")
        return {"overview": "", "main_themes": [], "key_takeaways": [], "tags": [], "_error": str(e)}
