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


_ENRICH_PROMPT = """Analyze these text chunks and return a JSON array with one object per chunk.
Each object must have exactly these fields:
- "title": Concise title (3-8 words)
- "summary": 1-2 sentence summary
- "tags": Array of 3-6 relevant tags (lowercase, no spaces, use hyphens)
- "questions": Array of 2-3 natural questions this text answers
- "semantic_key": 2-5 word subtopic identifier

Return ONLY a valid JSON array, no other text.

Chunks:
{chunks}"""


def _llm_call_with_retry(provider, messages, max_retries: int = 3) -> str:
    """LLM call with exponential backoff on rate limits."""
    for attempt in range(max_retries + 1):
        try:
            response = provider.chat(messages, model=None)
            if response is None:
                raise ValueError("Provider returned None")
            return response
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate" in err_str.lower()
            if is_rate_limit and attempt < max_retries:
                wait = 2 ** attempt * 3
                print(f"  [enrich] Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            raise


def enrich_llm(chunks: list[dict], provider, batch_size: int = 5, calls_per_min: float = 7.5) -> list[dict]:
    """Add LLM-generated enrichment: title, summary, tags, questions, semantic_key.

    Batches multiple chunks per LLM call for efficiency.
    Throttles to calls_per_min to avoid rate limits on free tiers.
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
        if not _run_batch(batch, f"Batch {batch_num}/{total_batches}"):
            failed_batches.append(batch)

    if failed_batches:
        print(f"  [enrich] Retrying {len(failed_batches)} failed batches...")
        still_failed = 0
        for i, batch in enumerate(failed_batches):
            if not _run_batch(batch, f"Retry {i+1}/{len(failed_batches)}"):
                still_failed += 1
        if still_failed:
            print(f"  [enrich] {still_failed} batches permanently failed")

    return chunks
