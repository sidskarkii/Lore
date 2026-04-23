import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LORE_DATA_DIR", tempfile.mkdtemp(prefix="lore-review-v2-"))
sys.path.insert(0, str(REPO_ROOT / "src"))
load_dotenv(REPO_ROOT / ".env")

from lore.core.chunk import chunk_sections
from lore.core.extractors import ExtractedDocument
from lore.core.ingest import Ingester


def test_archive_rollback_preserves_previous_final_on_tmp_rename_failure():
    ing = Ingester()
    doc = ExtractedDocument(
        sections=[{"title": "A", "text": "new body"}],
        metadata={"book_title": "T"},
        source_type="pdf",
        file_path="/tmp/x.pdf",
    )

    archive_dir = ing._cfg.archive_dir
    final = archive_dir / "archive_rollback_check"
    backup = archive_dir / ".archive_rollback_check.bak"
    tmp = archive_dir / ".archive_rollback_check.tmp"

    import shutil

    for path in (final, backup, tmp):
        if path.exists():
            shutil.rmtree(path)

    final.mkdir(parents=True)
    (final / "sentinel.txt").write_text("old-state")

    original_rename = Path.rename

    def selective_fail(self, target):
        if self == tmp and target == final:
            raise OSError("simulated tmp->final failure")
        return original_rename(self, target)

    Path.rename = selective_fail
    try:
        try:
            ing._save_archive(
                "archive_rollback_check",
                doc,
                [{"text": "chunk"}],
                {"collection": "archive_rollback_check"},
            )
        except OSError:
            pass
    finally:
        Path.rename = original_rename

    assert final.exists()
    assert (final / "sentinel.txt").read_text() == "old-state"
    assert not backup.exists()


def test_largest_untitled_block_should_still_win_page_and_chapter_metadata():
    chunks = chunk_sections(
        [
            {"title": "Intro", "text": " ".join(["a"] * 10), "page_num": 1, "chapter": "c1"},
            {"title": "", "text": " ".join(["b"] * 40), "page_num": 9, "chapter": "c9"},
        ],
        target_tokens=512,
        source_path="book.pdf",
    )

    assert len(chunks) == 1
    assert chunks[0]["page_num"] == 9
    assert chunks[0]["chapter"] == "c9"


def main():
    tests = [
        test_archive_rollback_preserves_previous_final_on_tmp_rename_failure,
        test_largest_untitled_block_should_still_win_page_and_chapter_metadata,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
