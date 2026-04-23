import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LORE_DATA_DIR", tempfile.mkdtemp(prefix="lore-review-unreviewed-"))
sys.path.insert(0, str(REPO_ROOT / "src"))
load_dotenv(REPO_ROOT / ".env")

from lore.core.database import Database
from lore.core.enrich import _extract_json
from lore.mcp.server import create_mcp_server


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_extract_json_edge_cases():
    cases = [
        ("```json\n[{\"a\": 1,},]\n```", [{"a": 1}]),
        ("Before\n```text\nignore me\n```\n```json\n{'a': 'b'}\n```\nAfter", [{"a": "b"}]),
        ("Some prose first {\"x\": 1, \"y\": 2,} trailing prose", [{"x": 1, "y": 2}]),
        ("```python\nx = {'bad': True}\n```\n```json\n[{\"ok\": true}]\n```", [{"ok": True}]),
    ]
    for raw, expected in cases:
        result = _extract_json(raw)
        check(result == expected, f"unexpected parse for {raw!r}: {result!r}")


def test_ttl_filtering_and_reset():
    db = Database(Path(tempfile.gettempdir()) / "lore_review_unreviewed.db")
    try:
        db._conn.execute("DELETE FROM interactions")
        db._conn.execute("DELETE FROM chunk_ratings")
        db._conn.commit()

        db.log_interaction("s1", "get_context", chunk_ids_fetched=["fresh"])
        db.log_interaction("s1", "search", query="alpha")
        db.log_interaction("s1", "search_deep", query="alpha")
        db.log_interaction("s1", "search", query="beta")
        db.rate_chunk("c1", True)
        db.log_interaction("s1", "get_context", chunk_ids_fetched=["old"])
        db._conn.execute(
            "UPDATE interactions SET timestamp = ? WHERE session_id = ? AND action = 'get_context' AND chunk_ids_fetched = ?",
            ("2000-01-01T00:00:00+00:00", "s1", json.dumps(["old"])),
        )
        db._conn.commit()

        all_ids = db.get_session_fetched_ids("s1")
        ttl_ids = db.get_session_fetched_ids("s1", ttl_minutes=30)
        check(all_ids == {"fresh", "old"}, f"unexpected all ids: {all_ids!r}")
        check(ttl_ids == {"fresh"}, f"unexpected ttl ids: {ttl_ids!r}")

        top_queries = db.get_top_queries(5)
        check(top_queries[0]["query"] == "alpha" and top_queries[0]["count"] == 2, f"unexpected top queries: {top_queries!r}")

        top_chunks = db.get_top_chunks(5)
        chunk_ids = {row["chunk_id"] for row in top_chunks}
        check({"fresh", "old", "c1"} <= chunk_ids, f"missing chunk ratings in top chunks: {top_chunks!r}")

        db.reset_session_fetched("s1")
        after_reset = db.get_session_fetched_ids("s1")
        check(after_reset == set(), f"reset should clear fetched ids, got {after_reset!r}")

        remaining = db._conn.execute(
            "SELECT action, query FROM interactions WHERE session_id = ? ORDER BY id",
            ("s1",),
        ).fetchall()
        remaining_actions = [row["action"] for row in remaining]
        check(remaining_actions == ["search", "search_deep", "search"], f"unexpected interactions after reset: {remaining_actions!r}")
    finally:
        db.close()
        try:
            Path(tempfile.gettempdir(), "lore_review_unreviewed.db").unlink()
        except FileNotFoundError:
            pass


def test_intro_shape():
    mcp = create_mcp_server()
    intro = mcp._tool_manager._tools["intro"].fn()
    check(intro["success"] is True, f"intro failed: {intro!r}")
    check("health" in intro, f"missing health info: {intro!r}")
    check("collections" in intro and isinstance(intro["collections"], list), f"missing collections list: {intro!r}")
    check("usage" in intro and isinstance(intro["usage"], dict), f"missing usage: {intro!r}")
    check("overview" in intro and "total_collections" in intro["overview"], f"missing overview counts: {intro!r}")


def main():
    tests = [
        test_extract_json_edge_cases,
        test_ttl_filtering_and_reset,
        test_intro_shape,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
