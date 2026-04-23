"""Batch ingest all books from ~/Documents/BOOKS/ into Lore."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from lore.core.ingest import Ingester

BOOKS_DIR = Path.home() / "Documents" / "BOOKS"

BOOKS = [
    # (filename_substring, display_name, topic, subtopic)

    # Strategy / Psychology
    ("Ego Is the Enemy", "Ego Is the Enemy", "psychology", "self-improvement"),
    ("48 Laws of Power", "The 48 Laws of Power", "strategy", "power"),
    ("Influence - The Psychology", "Influence: The Psychology of Persuasion", "psychology", "persuasion"),
    ("Laws of Human Nature", "The Laws of Human Nature", "psychology", "human-nature"),
    ("How to Analyze People", "How to Analyze People", "psychology", "body-language"),
    ("Social Engineering The Art", "Social Engineering", "psychology", "social-engineering"),
    ("Psychological Warfare Discover", "The Art of Psychological Warfare", "psychology", "warfare"),

    # AI / Technical
    ("LangChain, LangGraph, and MCP", "AI Agents with LangChain and MCP", "ai", "agents"),
    ("AI Agents with MCP (First", "AI Agents with MCP", "ai", "agents"),
    ("Build a DeepSeek", "Build a DeepSeek Model", "ai", "deep-learning"),
    ("Build a Multi-Agent System", "Build a Multi-Agent System with MCP", "ai", "agents"),
    ("Build a Reasoning Model", "Build a Reasoning Model", "ai", "deep-learning"),
    ("Build a Text-to-Image", "Build a Text-to-Image Generator", "ai", "diffusion"),
    ("Building Applications with AI", "Building Applications with AI Agents", "ai", "agents"),
    ("Machine Learning for Drug", "ML for Drug Discovery", "ai", "drug-discovery"),
    ("Quantum Computing in Action", "Quantum Computing in Action", "tech", "quantum"),
]


def find_file(substring: str) -> Path | None:
    if not BOOKS_DIR.is_dir():
        return None
    for f in sorted(BOOKS_DIR.iterdir(), key=lambda p: p.name):
        if substring in f.name:
            return f
    return None


def main():
    ingester = Ingester()
    total_books = len(BOOKS)
    total_chunks = 0
    failed = []

    for i, (substring, name, topic, subtopic) in enumerate(BOOKS, 1):
        filepath = find_file(substring)
        if not filepath:
            print(f"\n[{i}/{total_books}] SKIP — file not found: {substring}")
            failed.append((name, "file not found"))
            continue

        print(f"\n{'='*60}")
        print(f"[{i}/{total_books}] {name}")
        print(f"  File: {filepath.name}")
        print(f"  Topic: {topic}/{subtopic}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            n = ingester.ingest_file(
                path=str(filepath),
                name=name,
                topic=topic,
                subtopic=subtopic,
                enrich=True,
            )
            elapsed = time.time() - t0
            total_chunks += n
            print(f"\n  DONE: {n} chunks in {elapsed:.0f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  FAILED after {elapsed:.0f}s: {e}")
            failed.append((name, str(e)))

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {total_chunks} total chunks from {total_books - len(failed)}/{total_books} books")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for name, err in failed:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
