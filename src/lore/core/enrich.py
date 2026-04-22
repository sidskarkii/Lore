"""Chunk enrichment — programmatic (KeyBERT, spaCy) + LLM-based."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

_enrichment_cache: dict[str, dict] | None = None


def _get_enrichment_cache() -> dict[str, dict]:
    global _enrichment_cache
    if _enrichment_cache is None:
        cache_path = Path(__file__).resolve().parents[3] / ".enrichment_cache.json"
        if cache_path.exists():
            _enrichment_cache = json.loads(cache_path.read_text())
            print(f"  [enrich] Loaded {len(_enrichment_cache)} cached enrichments")
        else:
            _enrichment_cache = {}
    return _enrichment_cache


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _extract_json(response: str) -> list[dict]:
    """Extract JSON array from LLM response, handling common malformations."""
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

_kw_model = None
_nlp = None


def _get_kw_model():
    global _kw_model
    if _kw_model is None:
        from keybert import KeyBERT
        print("  [enrich] Loading KeyBERT model...")
        _kw_model = KeyBERT(model="all-MiniLM-L6-v2")
        print("  [enrich] KeyBERT ready.")
    return _kw_model


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
        print("  [enrich] spaCy ready.")
    return _nlp


def enrich_programmatic(chunks: list[dict]) -> list[dict]:
    """Add keywords and entities to chunks without LLM calls."""
    texts = [c["text"] for c in chunks]

    keywords_list = _extract_keywords_batch(texts)
    entities_list = _extract_entities_batch(texts)

    for chunk, kws, ents in zip(chunks, keywords_list, entities_list):
        chunk["keywords"] = ", ".join(kws)
        chunk["entities"] = json.dumps(ents) if ents else ""

    return chunks


def _extract_keywords_batch(texts: list[str]) -> list[list[str]]:
    """Extract keywords using KeyBERT."""
    try:
        kw_model = _get_kw_model()
        results = []
        for text in texts:
            if len(text.split()) < 10:
                results.append([])
                continue
            kws = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=6,
                use_mmr=True,
                diversity=0.5,
            )
            results.append([kw for kw, _ in kws])
        return results
    except ImportError:
        print("  KeyBERT not installed — skipping keyword extraction")
        return [[] for _ in texts]


def _extract_entities_batch(texts: list[str]) -> list[list[dict]]:
    """Extract named entities using spaCy."""
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


_CHUNK_ENRICH_PROMPT = """You are analyzing passages from "{book_title}"{section_context}.

For each passage, return a JSON array with one object per passage containing:
- "title": Concise title (3-8 words) capturing the main point
- "summary": 1-2 sentence summary of what this passage says
- "tags": Array of 3-6 tags (lowercase, hyphens, no spaces) for topic categorization
- "importance": Integer 1-5 rating how central this passage is to the chapter's argument (5=core thesis, 1=tangential)
- "semantic_key": 2-5 word subtopic identifier

Return ONLY a valid JSON array, no other text.

Passages:
{chunks}"""

_SECTION_SUMMARY_PROMPT = """You are reading a section of "{book_title}".
Section: {section_name}

Below is the full text of this section. Write a structured summary.

Return a JSON object with:
- "summary": 3-5 sentence summary capturing the key argument and conclusions
- "key_concepts": Array of 3-8 main concepts or ideas discussed
- "key_entities": Array of people, organizations, or works mentioned
- "importance_adjustments": Array of objects {{"chunk_index": N, "importance": 1-5}} for any passages whose importance you'd adjust now that you see the full section context

Return ONLY valid JSON, no other text.

Section text:
{section_text}"""

_SECTION_PROGRESSIVE_PROMPT = """You are reading "{book_title}", section "{section_name}".

Here is your running summary so far:
{running_summary}

Now read the next passages and update your summary.

Return a JSON object with:
- "running_summary": Updated 3-5 sentence summary incorporating the new material
- "key_concepts": Updated array of main concepts (add new ones, keep existing)
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


_FALLBACK_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-3-12b-it:free",
]


def _llm_call_with_retry(provider, messages, max_retries: int = 3) -> str:
    """LLM call with exponential backoff on rate limits + fallback on content moderation."""
    for attempt in range(max_retries + 1):
        try:
            response = provider.chat(messages, model=None)
            if response is None:
                raise ValueError("Provider returned None")
            return response
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate" in err_str.lower()
            is_moderation = "403" in err_str or "moderation" in err_str.lower() or "flagged" in err_str.lower()

            if is_moderation:
                print(f"  [enrich] Content moderation block, trying fallback models...")
                for fallback in _FALLBACK_MODELS:
                    try:
                        response = provider.chat(messages, model=fallback)
                        if response is None:
                            continue
                        print(f"  [enrich] Fallback succeeded: {fallback}")
                        return response
                    except Exception:
                        continue
                raise

            if is_rate_limit and attempt < max_retries:
                wait = 2 ** attempt * 3
                print(f"  [enrich] Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            raise


_MAX_TOKENS_PER_PASS = 5000


def enrich_chunks_stage2(
    chunks: list[dict],
    provider,
    book_title: str = "",
    calls_per_min: float = 7.5,
    on_progress=None,
) -> list[dict]:
    """Stage 2: Chunk-level titles, tags, importance. Context-aware."""
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
            cached = cache[h]
            for field in ("title", "summary", "keywords", "questions", "semantic_key"):
                if cached.get(field):
                    c[field] = cached[field]
            cached_count += 1
        else:
            enrichable.append((i, c))

    total_batches = (len(enrichable) + batch_size - 1) // batch_size
    print(f"  [stage2] Chunk enrichment: {cached_count} cached, {len(enrichable)} need LLM ({total_batches} batches)")

    last_call = 0.0
    failed: list[list[tuple[int, dict]]] = []

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

        try:
            last_call = time.time()
            prompt = _CHUNK_ENRICH_PROMPT.format(
                book_title=book_title or "Unknown",
                section_context=section_ctx,
                chunks=chunks_text,
            )
            response = _llm_call_with_retry(provider, [{"role": "user", "content": prompt}])
            results = _extract_json(response)

            for (orig_idx, chunk), enrichment in zip(batch, results):
                chunk["title"] = enrichment.get("title", "")
                chunk["summary"] = enrichment.get("summary", "")
                chunk["keywords"] = ", ".join(enrichment.get("tags", []))
                chunk["importance"] = int(enrichment.get("importance", 3))
                chunk["semantic_key"] = enrichment.get("semantic_key", "")
        except Exception as e:
            print(f"  [stage2] Batch {batch_num} failed: {e}")
            failed.append(batch)

    if failed:
        print(f"  [stage2] Retrying {len(failed)} failed batches...")
        for batch in failed:
            elapsed = time.time() - last_call
            if last_call > 0 and elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            try:
                last_call = time.time()
                chunks_text = "\n".join(f"--- Passage {i+1} ---\n{c['text']}" for i, (_, c) in enumerate(batch))
                prompt = _CHUNK_ENRICH_PROMPT.format(book_title=book_title or "Unknown", section_context="", chunks=chunks_text)
                response = _llm_call_with_retry(provider, [{"role": "user", "content": prompt}])
                results = _extract_json(response)
                for (orig_idx, chunk), enrichment in zip(batch, results):
                    chunk["title"] = enrichment.get("title", "")
                    chunk["summary"] = enrichment.get("summary", "")
                    chunk["keywords"] = ", ".join(enrichment.get("tags", []))
                    chunk["importance"] = int(enrichment.get("importance", 3))
                    chunk["semantic_key"] = enrichment.get("semantic_key", "")
            except Exception as e:
                print(f"  [stage2] Retry failed: {e}")

    return chunks


def enrich_section_stage3(
    section_chunks: list[dict],
    provider,
    book_title: str = "",
    section_name: str = "",
    calls_per_min: float = 7.5,
) -> dict:
    """Stage 3: Section/chapter summary via progressive passes over original text."""
    min_interval = 60.0 / calls_per_min

    total_tokens = sum(len(c.get("text", "")) // 4 for c in section_chunks)

    if total_tokens <= _MAX_TOKENS_PER_PASS:
        section_text = "\n\n".join(c["text"] for c in section_chunks)
        prompt = _SECTION_SUMMARY_PROMPT.format(
            book_title=book_title or "Unknown",
            section_name=section_name,
            section_text=section_text,
        )
        try:
            response = _llm_call_with_retry(provider, [{"role": "user", "content": prompt}])
            return _extract_json(response) if isinstance(_extract_json(response), dict) else _extract_json(response)[0]
        except Exception as e:
            print(f"  [stage3] Section summary failed: {e}")
            return {"summary": "", "key_concepts": [], "key_entities": []}

    running_summary = "No summary yet — this is the first pass."
    chunk_titles = []

    tokens_so_far = 0
    batch_chunks = []
    for c in section_chunks:
        ct = len(c.get("text", "")) // 4
        if tokens_so_far + ct > _MAX_TOKENS_PER_PASS and batch_chunks:
            chunks_text = "\n\n".join(bc["text"] for bc in batch_chunks)
            prompt = _SECTION_PROGRESSIVE_PROMPT.format(
                book_title=book_title or "Unknown",
                section_name=section_name,
                running_summary=running_summary,
                chunks=chunks_text,
            )
            try:
                time.sleep(min_interval)
                response = _llm_call_with_retry(provider, [{"role": "user", "content": prompt}])
                result = _extract_json(response)
                if isinstance(result, list):
                    result = result[0]
                running_summary = result.get("running_summary", running_summary)
                chunk_titles.extend(result.get("chunk_titles", []))
            except Exception as e:
                print(f"  [stage3] Progressive pass failed: {e}")

            batch_chunks = []
            tokens_so_far = 0

        batch_chunks.append(c)
        tokens_so_far += ct

    if batch_chunks:
        chunks_text = "\n\n".join(bc["text"] for bc in batch_chunks)
        prompt = _SECTION_PROGRESSIVE_PROMPT.format(
            book_title=book_title or "Unknown",
            section_name=section_name,
            running_summary=running_summary,
            chunks=chunks_text,
        )
        try:
            time.sleep(min_interval)
            response = _llm_call_with_retry(provider, [{"role": "user", "content": prompt}])
            result = _extract_json(response)
            if isinstance(result, list):
                result = result[0]
            running_summary = result.get("running_summary", running_summary)
            chunk_titles.extend(result.get("chunk_titles", []))
        except Exception as e:
            print(f"  [stage3] Final pass failed: {e}")

    return {
        "summary": running_summary,
        "key_concepts": [],
        "key_entities": [],
        "chunk_titles": chunk_titles,
    }


def enrich_book_stage4(
    section_summaries: list[dict],
    provider,
    book_title: str = "",
    author: str = "",
    toc: list[str] | None = None,
) -> dict:
    """Stage 4: Book-level summary from section summaries."""
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
        print(f"  [stage4] Book summary failed: {e}")
        return {"overview": "", "main_themes": [], "key_takeaways": [], "tags": []}


def enrich_llm(chunks: list[dict], provider, batch_size: int = 5, calls_per_min: float = 7.5, on_progress=None) -> list[dict]:
    """Legacy wrapper — runs stage 2 chunk enrichment. Use enrich_chunks_stage2 for new code.

    Batches multiple chunks per LLM call for efficiency.
    Throttles to calls_per_min to avoid rate limits on free tiers.
    on_progress: optional callback(batch_num, total_batches, cached_count) for live status.
    """
    min_interval = 60.0 / calls_per_min
    cache = _get_enrichment_cache()

    enrichable = []
    cached_count = 0
    for i, c in enumerate(chunks):
        if len(c["text"].split()) < 15:
            continue
        h = _content_hash(c["text"])
        if h in cache:
            cached = cache[h]
            c["title"] = cached.get("title", "")
            c["summary"] = cached.get("summary", "")
            c["keywords"] = cached.get("keywords", c.get("keywords", ""))
            c["questions"] = cached.get("questions", "")
            c["semantic_key"] = cached.get("semantic_key", "")
            cached_count += 1
        else:
            enrichable.append((i, c))

    total_batches = (len(enrichable) + batch_size - 1) // batch_size
    print(f"  [enrich] LLM enrichment: {cached_count} from cache, {len(enrichable)} need LLM ({total_batches} batches)")

    failed_batches: list[list[tuple[int, dict]]] = []
    last_call = 0.0

    def _run_batch(batch: list[tuple[int, dict]], batch_label: str) -> bool:
        nonlocal last_call
        elapsed = time.time() - last_call
        if last_call > 0 and elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        print(f"  [enrich] {batch_label}...")

        chunks_text = ""
        for idx, (_, chunk) in enumerate(batch):
            chunks_text += f"\n--- Chunk {idx + 1} ---\n{chunk['text']}\n"

        try:
            last_call = time.time()
            prompt = _ENRICH_PROMPT.format(chunks=chunks_text)
            response = _llm_call_with_retry(provider, [{"role": "user", "content": prompt}])
            results = _extract_json(response)

            for (orig_idx, chunk), enrichment in zip(batch, results):
                chunk["title"] = enrichment.get("title", "")
                chunk["summary"] = enrichment.get("summary", "")
                chunk["keywords"] = ", ".join(enrichment.get("tags", []))
                chunk["questions"] = json.dumps(enrichment.get("questions", []))
                chunk["semantic_key"] = enrichment.get("semantic_key", "")
            return True
        except Exception as e:
            print(f"  [enrich] Failed: {e}")
            return False

    for batch_start in range(0, len(enrichable), batch_size):
        batch = enrichable[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        if on_progress:
            on_progress(batch_num, total_batches, cached_count)
        if not _run_batch(batch, f"Batch {batch_num}/{total_batches}"):
            failed_batches.append(batch)

    if failed_batches:
        print(f"  [enrich] Retrying {len(failed_batches)} failed batches...")
        still_failed = 0
        for i, batch in enumerate(failed_batches):
            if on_progress:
                on_progress(total_batches + i + 1, total_batches + len(failed_batches), cached_count)
            if not _run_batch(batch, f"Retry {i+1}/{len(failed_batches)}"):
                still_failed += 1
        if still_failed:
            print(f"  [enrich] {still_failed} batches permanently failed")

    return chunks
