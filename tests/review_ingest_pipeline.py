import json
import os
import sys
import tempfile
import threading
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LORE_DATA_DIR", tempfile.mkdtemp(prefix="lore-review-"))
sys.path.insert(0, str(REPO_ROOT / "src"))
load_dotenv(REPO_ROOT / ".env")

from lore.core.chunk import chunk_sections
from lore.core.enrich import enrich_section_stage3
from lore.core.extractors import ExtractedDocument
from lore.core.ingest import Ingester
from lore.core.store import Store
import lore.core.enrich as enrich_mod
import lore.core.store as store_mod


class FakeStore:
    def __init__(self, barrier: threading.Barrier | None = None):
        self.barrier = barrier
        self.add_calls = 0
        self.calls = []
        self._lock = threading.Lock()

    def list_collections(self):
        if self.barrier is not None:
            self.barrier.wait()
        return []

    def add_chunks(self, chunks, meta):
        with self._lock:
            self.add_calls += 1
            self.calls.append((chunks, meta))
        return len(chunks)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_chunk_section_metadata_smear():
    chunks = chunk_sections(
        [
            {"title": "Intro", "text": " ".join(["a"] * 30), "page_num": 1},
            {"title": "Real Section", "text": " ".join(["b"] * 200), "page_num": 25},
        ],
        target_tokens=128,
        source_path="book.pdf",
    )
    check(len(chunks) >= 1, "expected at least one chunk")
    first = chunks[0]
    check("Real Section" in first["text"], "expected merged text to include the second section")
    check(first["section_heading"] == "Real Section", f"expected larger section metadata after merge fix; got {first['section_heading']!r}")
    check(first["page_num"] == 25, f"expected larger section page after merge fix; got {first['page_num']!r}")


def test_archive_partial_write_on_failure():
    ing = Ingester(store=FakeStore())
    doc = ExtractedDocument(
        sections=[{"title": "Only", "text": "body text"}],
        metadata={"book_title": "Archive Test"},
        source_type="pdf",
        file_path="/tmp/archive-test.pdf",
    )
    archive_dir = ing._cfg.archive_dir / "archive_partial_write"
    if archive_dir.exists():
        for child in archive_dir.iterdir():
            child.unlink()
        archive_dir.rmdir()

    original_write_text = Path.write_text

    def flaky_write_text(self, data, *args, **kwargs):
        if self.name == "chunks.json":
            raise OSError("simulated disk failure")
        return original_write_text(self, data, *args, **kwargs)

    Path.write_text = flaky_write_text
    try:
        try:
            ing._save_archive(
                "archive_partial_write",
                doc,
                [{"text": "chunk body"}],
                {"collection": "archive_partial_write"},
                section_summaries=[{"section": "Only", "summary": "x"}],
                book_summary={"overview": "y"},
            )
        except OSError:
            pass
    finally:
        Path.write_text = original_write_text

    check(not archive_dir.exists(), "final archive dir should NOT exist after atomic write failure (writes go to .tmp first)")
    tmp_dir = ing._cfg.archive_dir / ".archive_partial_write.tmp"
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir)


def test_same_collection_race_prevented():
    store = FakeStore()
    ing = Ingester(store=store)
    ing._save_archive = lambda *args, **kwargs: None

    doc = ExtractedDocument(
        sections=[{"title": "Only", "text": " ".join(["body"] * 120)}],
        metadata={"book_title": "Race Book"},
        source_type="pdf",
        file_path="/tmp/race.pdf",
    )

    results = []

    def worker():
        try:
            n = ing._ingest_extracted(
                doc=doc,
                name="Race Book",
                topic="topic",
                subtopic="subtopic",
                source_path="/tmp/race.pdf",
                enrich=False,
            )
            results.append(n)
        except Exception as exc:
            results.append(exc)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    adds = [r for r in results if isinstance(r, int) and r > 0]
    skips = [r for r in results if isinstance(r, int) and r == 0]
    check(len(adds) == 1, f"expected exactly one ingest to succeed, got {len(adds)} adds")
    check(len(skips) == 1, f"expected one ingest to skip, got {len(skips)} skips")


def test_stage3_fallback_drops_recovered_summary():
    original_retry = enrich_mod._llm_call_with_retry
    original_fallback = enrich_mod._llm_call_with_fallback
    calls = {"retry": 0}

    def fail_retry(provider, messages, max_retries=2):
        calls["retry"] += 1
        raise RuntimeError("primary model failed")

    def fallback(provider, messages):
        return json.dumps(
            {
                "running_summary": "Recovered summary from fallback",
                "key_concepts": ["concept-a"],
                "key_entities": ["entity-a"],
                "chunk_titles": [{"title": "Recovered title", "importance": 4}],
            }
        )

    enrich_mod._llm_call_with_retry = fail_retry
    enrich_mod._llm_call_with_fallback = fallback
    try:
        result = enrich_section_stage3(
            section_chunks=[{"text": " ".join(["x"] * 80)}],
            provider=object(),
            book_title="Fallback Book",
            section_name="Section 1",
        )
    finally:
        enrich_mod._llm_call_with_retry = original_retry
        enrich_mod._llm_call_with_fallback = original_fallback

    check(result["summary"] == "Recovered summary from fallback", f"fallback should now recover summary: {result['summary']!r}")
    check(result["key_concepts"] == ["concept-a"], f"fallback concepts should still be captured: {result['key_concepts']!r}")


def test_store_partial_overlap_rerun_loses_existing_chunks():
    class FakeQuery:
        def __init__(self, rows):
            self.rows = rows

        def where(self, expr):
            return self

        def select(self, cols):
            return FakeQuery([{col: row.get(col) for col in cols} for row in self.rows])

        def limit(self, n):
            return FakeQuery(self.rows[:n])

        def to_list(self):
            return list(self.rows)

    class FakeTable:
        def __init__(self):
            self.rows = [
                {"collection": "demo", "episode_num": 1, "content_hash": "old-a00000000000", "id": "a"},
                {"collection": "demo", "episode_num": 1, "content_hash": "old-b00000000000", "id": "b"},
            ]

        def search(self):
            return FakeQuery(self.rows)

        def delete(self, expr):
            self.rows = [r for r in self.rows if not (r["collection"] == "demo" and r["episode_num"] == 1)]

        def add(self, rows):
            self.rows.extend(rows)

    fake_table = FakeTable()
    store = Store.__new__(Store)
    store._dim = 1
    store._table_name = "chunks"
    store._table = fake_table
    store._db = type("DB", (), {"open_table": lambda self, name: fake_table})()
    store._get_or_create_table = lambda: fake_table
    store._rebuild_fts = lambda tbl: None
    store._optimize = lambda tbl: None

    original_embed_texts = store_mod.embed_texts
    original_hashlib_sha256 = store_mod.hashlib.sha256

    hash_map = {
        "shared text": "old-a00000000000",
        "new text": "new-c00000000000",
    }

    class FakeHash:
        def __init__(self, value):
            self.value = value

        def hexdigest(self):
            return self.value.ljust(64, "0")

    def fake_sha256(data):
        text = data.decode() if isinstance(data, bytes) else data
        return FakeHash(hash_map[text])

    store_mod.embed_texts = lambda texts: [[0.0] for _ in texts]
    store_mod.hashlib.sha256 = fake_sha256
    try:
        added = store.add_chunks(
            chunks=[
                {"text": "shared text", "start_sec": 0, "end_sec": 0},
                {"text": "new text", "start_sec": 0, "end_sec": 0},
            ],
            meta={
                "collection": "demo",
                "collection_display": "Demo",
                "topic": "",
                "subtopic": "",
                "episode_num": 1,
                "episode_title": "Episode",
                "url": "",
                "source_type": "video",
            },
        )
    finally:
        store_mod.embed_texts = original_embed_texts
        store_mod.hashlib.sha256 = original_hashlib_sha256

    check(added == 1, f"expected only one chunk to be considered new, got {added}")
    hashes_after = {row["content_hash"] for row in fake_table.rows}
    check(hashes_after == {"new-c00000000000"}, f"expected old overlapping chunk to be lost after delete/add, got {hashes_after!r}")


def main():
    tests = [
        test_chunk_section_metadata_smear,
        test_archive_partial_write_on_failure,
        test_same_collection_race_prevented,
        test_stage3_fallback_drops_recovered_summary,
        test_store_partial_overlap_rerun_loses_existing_chunks,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
