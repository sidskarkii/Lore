"""Chunking — group segments and text into overlapping windows."""

from __future__ import annotations
import math

from .config import get_config


def chunk_segments(
    segments: list[dict],
    target_sec: float | None = None,
    overlap_sec: float | None = None,
) -> list[dict]:
    """Group whisper segments into overlapping ~target_sec windows.

    Each segment must have: {"start": float, "end": float, "text": str}
    Returns: [{"text": str, "start_sec": float, "end_sec": float}, ...]
    """
    if not segments:
        return []

    cfg = get_config()
    if target_sec is None:
        target_sec = cfg.get("chunking.target_sec", 90)
    if overlap_sec is None:
        overlap_sec = cfg.get("chunking.overlap_sec", 15)

    chunks: list[dict] = []
    buf_segs: list[dict] = []
    buf_start: float | None = None

    for seg in segments:
        if buf_start is None:
            buf_start = seg["start"]
        buf_segs.append(seg)

        if seg["end"] - buf_start >= target_sec:
            text = " ".join(s["text"] for s in buf_segs)
            chunks.append({
                "text": text,
                "start_sec": buf_start,
                "end_sec": seg["end"],
            })

            # Keep last overlap_sec worth of segments for next chunk
            overlap: list[dict] = []
            acc = 0.0
            for s in reversed(buf_segs):
                overlap.insert(0, s)
                acc += s["end"] - s["start"]
                if acc >= overlap_sec:
                    break
            buf_segs = overlap
            buf_start = buf_segs[0]["start"] if buf_segs else None

    # Flush remaining
    if buf_segs:
        text = " ".join(s["text"] for s in buf_segs)
        chunks.append({
            "text": text,
            "start_sec": buf_start,
            "end_sec": buf_segs[-1]["end"],
        })

    return chunks


def chunk_text(
    text: str,
    target_tokens: int = 512,
    overlap_tokens: int = 0,
    source_path: str = "",
) -> list[dict]:
    """Chunk plain text into segments using recursive word-based splitting.

    Approximates tokens as ``words / 0.75`` (≈1.33 tokens per word).

    Returns list of ``{"text": str, "start_sec": 0, "end_sec": 0,
    "file_path": str}`` — ``start_sec``/``end_sec`` are always 0 because
    plain-text sources have no timestamps.
    """
    text = text.strip()
    if not text:
        return []

    words = text.split()
    target_words = max(1, int(target_tokens * 0.75))
    overlap_words = max(0, int(overlap_tokens * 0.75))

    # Fits in one chunk — return as-is.
    if len(words) <= target_words:
        return [{"text": text, "start_sec": 0, "end_sec": 0,
                 "file_path": source_path}]

    chunks: list[dict] = []
    start = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        chunk_words = words[start:end]
        chunks.append({
            "text": " ".join(chunk_words),
            "start_sec": 0,
            "end_sec": 0,
            "file_path": source_path,
        })
        step = target_words - overlap_words
        start += max(1, step)

    return chunks


def chunk_sections(
    sections: list[dict],
    target_tokens: int = 512,
    source_path: str = "",
) -> list[dict]:
    """Chunk pre-split sections from extractors.

    *Input*: ``[{"title": "Section Name", "text": "section content…"}, …]``

    Behaviour:
    - Empty sections (no text after stripping) are dropped.
    - Sections shorter than ``target_tokens / 3`` are merged with subsequent
      sections to avoid tiny fragments.
    - Sections exceeding *target_tokens* are split via :func:`chunk_text`.
    - Each section's title (when present) is prepended to its text.
    - Section titles are preserved as ``section_heading`` in chunk metadata.
    """
    target_words = max(1, int(target_tokens * 0.75))
    merge_threshold = target_words // 3

    prepared: list[tuple[str, str]] = []
    for sec in sections:
        body = (sec.get("text") or "").strip()
        title = (sec.get("title") or "").strip()
        page_num = sec.get("page_num", 0)
        chapter = sec.get("chapter", "")
        if not body:
            continue
        block = f"{title}\n{body}" if title else body
        prepared.append((block, title, page_num, chapter))

    if not prepared:
        return []

    merged: list[tuple[str, str, int, str]] = []
    buf_text = ""
    buf_heading = ""
    buf_page = 0
    buf_chapter = ""
    for block, heading, page_num, chapter in prepared:
        if buf_text:
            if len(buf_text.split()) < merge_threshold:
                buf_text = f"{buf_text}\n{block}"
                continue
            else:
                merged.append((buf_text, buf_heading, buf_page, buf_chapter))
                buf_text = block
                buf_heading = heading
                buf_page = page_num
                buf_chapter = chapter
        else:
            buf_text = block
            buf_heading = heading
            buf_page = page_num
            buf_chapter = chapter
    if buf_text:
        merged.append((buf_text, buf_heading, buf_page, buf_chapter))

    chunks: list[dict] = []
    for block, heading, page_num, chapter in merged:
        base_meta = {
            "start_sec": 0,
            "end_sec": 0,
            "file_path": source_path,
            "section_heading": heading,
            "page_num": page_num,
            "chapter": chapter,
        }
        word_count = len(block.split())
        if word_count <= target_words:
            chunks.append({"text": block, **base_meta})
        else:
            sub_chunks = chunk_text(block, target_tokens=target_tokens, source_path=source_path)
            for sc in sub_chunks:
                sc["section_heading"] = heading
                sc["page_num"] = page_num
                sc["chapter"] = chapter
            chunks.extend(sub_chunks)

    return chunks


def fmt_timestamp(secs: float) -> str:
    """Format seconds as MM:SS."""
    return f"{int(secs // 60):02}:{int(secs % 60):02}"
