"""Audio/video extraction — wraps existing Transcriber."""

from __future__ import annotations

from pathlib import Path

from . import ExtractedDocument


def extract_audio(path: str) -> ExtractedDocument:
    """Transcribe audio/video file and return as sections."""
    from ..transcribe import Transcriber

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    t = Transcriber()
    segments = t.transcribe(str(p))

    # Group every ~10 segments into a section
    sections: list[dict] = []
    group: list[dict] = []

    for seg in segments:
        group.append(seg)
        if len(group) >= 10:
            sections.append(_group_to_section(group))
            group = []

    if group:
        sections.append(_group_to_section(group))

    return ExtractedDocument(
        sections=sections,
        metadata={"duration_sec": segments[-1]["end"] if segments else 0},
        source_type="audio",
        file_path=str(p),
    )


def _group_to_section(segments: list[dict]) -> dict:
    """Merge a group of transcription segments into a section."""
    text = " ".join(s["text"] for s in segments)
    start = segments[0]["start"]
    end = segments[-1]["end"]
    return {
        "title": f"[{int(start // 60):02d}:{int(start % 60):02d} - {int(end // 60):02d}:{int(end % 60):02d}]",
        "text": text,
        "start_sec": start,
        "end_sec": end,
    }
