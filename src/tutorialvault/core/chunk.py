"""Chunking — group transcript segments into overlapping time windows."""

from __future__ import annotations

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


def fmt_timestamp(secs: float) -> str:
    """Format seconds as MM:SS."""
    return f"{int(secs // 60):02}:{int(secs % 60):02}"
