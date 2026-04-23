import json
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LORE_DATA_DIR", tempfile.mkdtemp(prefix="lore-enrich-review-"))
sys.path.insert(0, str(REPO_ROOT / "src"))
load_dotenv(REPO_ROOT / ".env")

import lore.core.enrich as enrich_mod


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_stage2_batch_uses_only_first_chunk_stage1_cues():
    original_retry = enrich_mod._llm_call_with_retry
    original_save = enrich_mod._save_enrichment_cache
    original_cache = enrich_mod._enrichment_cache
    captured = {}

    def fake_retry(provider, messages, max_retries=2):
        captured["messages"] = messages
        return json.dumps([
            {
                "title": "Chunk One",
                "summary": "Summary one.",
                "tags": ["alpha"],
                "concept_tags": ["concept-a"],
                "related_tags": [],
                "importance": 3,
                "why_important": "one",
                "questions": ["q1", "q2"],
                "self_contained": True,
                "confidence": "fact",
                "semantic_key": "key one",
            },
            {
                "title": "Chunk Two",
                "summary": "Summary two.",
                "tags": ["beta"],
                "concept_tags": ["concept-b"],
                "related_tags": [],
                "importance": 3,
                "why_important": "two",
                "questions": ["q3", "q4"],
                "self_contained": True,
                "confidence": "fact",
                "semantic_key": "key two",
            },
        ])

    enrich_mod._llm_call_with_retry = fake_retry
    enrich_mod._save_enrichment_cache = lambda: None
    enrich_mod._enrichment_cache = {}
    try:
        enrich_mod.enrich_chunks_stage2(
            [
                {"text": " ".join(["one"] * 40), "keywords": "alpha-kw", "entities": '[{"name":"Alice"}]'},
                {"text": " ".join(["two"] * 40), "keywords": "beta-kw", "entities": '[{"name":"Bob"}]'},
            ],
            provider=object(),
            book_title="Cue Test",
            calls_per_min=1000,
        )
    finally:
        enrich_mod._llm_call_with_retry = original_retry
        enrich_mod._save_enrichment_cache = original_save
        enrich_mod._enrichment_cache = original_cache

    prompt = captured["messages"][1]["content"]
    check("Keywords: alpha-kw" in prompt, "expected first chunk keywords in prompt")
    check("beta-kw" not in prompt, "second chunk keywords should not be missing from a multi-passage prompt")
    check("Bob" not in prompt, "second chunk entities should not be missing from a multi-passage prompt")


def test_stage3_first_pass_leaks_future_concepts():
    original_retry = enrich_mod._llm_call_with_retry
    original_max = enrich_mod._MAX_TOKENS_PER_PASS
    prompts = []

    def fake_retry(provider, messages, max_retries=2):
        prompts.append(messages[1]["content"])
        return json.dumps({
            "section_summary": "ok",
            "section_themes": [],
            "key_entities": [],
            "ledger_updates": [],
            "notable_points": [],
            "open_questions_or_tensions": [],
        })

    enrich_mod._llm_call_with_retry = fake_retry
    enrich_mod._MAX_TOKENS_PER_PASS = 30
    try:
        enrich_mod.enrich_section_stage3(
            [
                {"text": "a" * 80, "concept_tags": "alpha"},
                {"text": "b" * 80, "concept_tags": "beta"},
            ],
            provider=object(),
            book_title="Leak Test",
            section_name="S1",
            calls_per_min=1000,
        )
    finally:
        enrich_mod._llm_call_with_retry = original_retry
        enrich_mod._MAX_TOKENS_PER_PASS = original_max

    first_prompt = prompts[0]
    check("- beta (1x)" in first_prompt, "future chunk concept should not appear in first-pass ledger")


def test_stage4_concept_summary_counts_mentions_not_sections():
    original_retry = enrich_mod._llm_call_with_retry
    captured = {}

    def fake_retry(provider, messages, max_retries=2):
        captured["prompt"] = messages[0]["content"]
        return json.dumps({
            "overview": "ok",
            "main_themes": [],
            "key_takeaways": [],
            "tags": [],
            "cross_section_patterns": [],
        })

    enrich_mod._llm_call_with_retry = fake_retry
    try:
        enrich_mod.enrich_book_stage4(
            [
                {"section": "One", "summary": "s1", "concept_ledger": {"alpha": 5}},
                {"section": "Two", "summary": "s2", "concept_ledger": {"beta": 1}},
            ],
            provider=object(),
            book_title="Count Test",
        )
    finally:
        enrich_mod._llm_call_with_retry = original_retry

    check("- alpha (across 5 sections)" in captured["prompt"], "prompt should reveal current section-count wording bug")


def test_apply_cache_drops_false_boolean_fields():
    chunk = {}
    enrich_mod._apply_cache(chunk, {
        "title": "Cached",
        "summary": "Summary",
        "questions": "q1, q2",
        "self_contained": False,
    })
    check("self_contained" not in chunk, "false boolean cache values are currently skipped")


def test_stage3_fallback_drops_ledger_updates_and_tensions():
    original_retry = enrich_mod._llm_call_with_retry
    original_fallback = enrich_mod._llm_call_with_fallback

    def fail_retry(provider, messages, max_retries=2):
        raise RuntimeError("primary failed")

    def fallback(provider, messages):
        return json.dumps({
            "section_summary": "Recovered",
            "section_themes": [],
            "key_entities": [],
            "ledger_updates": [{"concept_tag": "fallback-concept", "action": "add", "evidence": "seen"}],
            "notable_points": [],
            "open_questions_or_tensions": ["open tension"],
        })

    enrich_mod._llm_call_with_retry = fail_retry
    enrich_mod._llm_call_with_fallback = fallback
    try:
        result = enrich_mod.enrich_section_stage3(
            [{"text": " ".join(["x"] * 80), "concept_tags": ""}],
            provider=object(),
            book_title="Fallback Loss",
            section_name="S1",
            calls_per_min=1000,
        )
    finally:
        enrich_mod._llm_call_with_retry = original_retry
        enrich_mod._llm_call_with_fallback = original_fallback

    check("fallback-concept" not in result["key_concepts"], "fallback ledger updates are currently ignored")
    check(result["tensions"] == [], "fallback tensions are currently ignored")


def main():
    tests = [
        test_stage2_batch_uses_only_first_chunk_stage1_cues,
        test_stage3_first_pass_leaks_future_concepts,
        test_stage4_concept_summary_counts_mentions_not_sections,
        test_apply_cache_drops_false_boolean_fields,
        test_stage3_fallback_drops_ledger_updates_and_tensions,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
