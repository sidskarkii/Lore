"""Chunk enrichment — programmatic (KeyBERT, spaCy) + LLM-based."""

from __future__ import annotations

import json


def enrich_programmatic(chunks: list[dict]) -> list[dict]:
    """Add keywords and entities to chunks without LLM calls.

    Modifies chunks in-place and returns them.
    """
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
        from keybert import KeyBERT

        kw_model = KeyBERT(model="all-MiniLM-L6-v2")
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
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

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


def enrich_llm(chunks: list[dict], provider) -> list[dict]:
    """Add LLM-generated enrichment: title, summary, questions, semantic_key.

    Uses MDKeyChunker-style single-call extraction for efficiency.
    Modifies chunks in-place and returns them.
    """
    prompt_template = """Analyze this text chunk and return a JSON object with exactly these fields:
- "title": A concise title (3-8 words)
- "summary": A 1-2 sentence summary of the key information
- "questions": An array of 2-3 natural questions this text answers
- "semantic_key": A 2-5 word subtopic identifier

Text:
---
{text}
---

Return ONLY valid JSON, no other text."""

    for chunk in chunks:
        text = chunk["text"]
        if len(text.split()) < 15:
            continue

        if len(text) > 3000:
            text = text[:3000] + "..."

        try:
            prompt = prompt_template.format(text=text)
            response = provider.chat(
                [{"role": "user", "content": prompt}],
                model=None,
            )

            json_str = response.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]

            data = json.loads(json_str)
            chunk["title"] = data.get("title", "")
            chunk["summary"] = data.get("summary", "")
            chunk["questions"] = json.dumps(data.get("questions", []))
            chunk["semantic_key"] = data.get("semantic_key", "")
        except Exception as e:
            print(f"  LLM enrichment failed for chunk: {e}")
            continue

    return chunks
