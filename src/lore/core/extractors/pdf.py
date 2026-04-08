"""PDF extractor — PyMuPDF4LLM with Marker/Surya OCR fallback."""

from __future__ import annotations

import re
from pathlib import Path

from . import ExtractedDocument

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^(#{1,3})\s+(.*)", re.MULTILINE)


def _split_markdown_sections(md: str) -> list[dict]:
    """Split markdown text by headers (# / ## / ###).

    Returns a list of ``{"title": str, "text": str}`` dicts.
    If no headers are found the entire text is returned as a single section.
    """
    matches = list(_HEADER_RE.finditer(md))

    if not matches:
        text = md.strip()
        return [{"title": "(untitled)", "text": text}] if text else []

    sections: list[dict] = []

    # Text before the first header (preamble)
    preamble = md[: matches[0].start()].strip()
    if preamble:
        sections.append({"title": "(preamble)", "text": preamble})

    for idx, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(md)
        body = md[start:end].strip()
        if body:
            sections.append({"title": title, "text": body})

    return sections


# ---------------------------------------------------------------------------
# Primary extractor — PyMuPDF4LLM
# ---------------------------------------------------------------------------

def _extract_with_pymupdf4llm(path: str) -> str | None:
    """Return markdown string or *None* on import/runtime failure."""
    try:
        import pymupdf4llm  # heavy optional dep — lazy load OK
    except ImportError:
        print("[pdf] pymupdf4llm not installed — skipping primary extractor")
        return None

    return pymupdf4llm.to_markdown(path)


# ---------------------------------------------------------------------------
# Fallback extractor — Marker (OCR)
# ---------------------------------------------------------------------------

def _extract_with_marker(path: str) -> str | None:
    """Return markdown string via Marker with force_ocr, or *None*."""
    try:
        from marker.converters.pdf import PdfConverter  # type: ignore[import]
        from marker.models import create_model_dict  # type: ignore[import]
    except ImportError:
        print("[pdf] marker not installed — OCR fallback unavailable")
        return None

    try:
        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(path)
        return rendered.markdown
    except Exception as exc:  # noqa: BLE001
        print(f"[pdf] marker extraction failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_MIN_CHARS = 100  # threshold below which we consider extraction "too little"


def extract_pdf(path: str) -> ExtractedDocument:
    """Extract text from a PDF file.

    Strategy:
    1. Try **PyMuPDF4LLM** (fast, native text).
    2. If the result is too short (<100 chars), fall back to **Marker/Surya**
       with OCR for scanned documents.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    md = _extract_with_pymupdf4llm(path)

    if md is None or len(md.strip()) < _MIN_CHARS:
        marker_md = _extract_with_marker(path)
        if marker_md and len(marker_md.strip()) >= _MIN_CHARS:
            md = marker_md

    if not md or not md.strip():
        md = ""

    sections = _split_markdown_sections(md)

    metadata = {
        "filename": p.name,
        "size_bytes": p.stat().st_size,
        "extractor": "pymupdf4llm",
    }

    return ExtractedDocument(
        sections=sections,
        metadata=metadata,
        source_type="pdf",
        file_path=path,
    )
