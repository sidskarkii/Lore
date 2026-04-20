"""PDF extractor — font-aware extraction with heading and code block detection."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from . import ExtractedDocument

_HEADER_RE = re.compile(r"^(#{1,3})\s+(.*)", re.MULTILINE)


def _is_mono(span: dict) -> bool:
    """Check if a span uses a monospace font."""
    if span["flags"] & 8:
        return True
    font_lower = span["font"].lower()
    return "mono" in font_lower or "courier" in font_lower or "consolas" in font_lower


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16)


def _extract_pages(path: str) -> str:
    """Extract full PDF with font-aware heading and code block detection."""
    doc = pymupdf.open(path)
    output: list[str] = []

    for page in doc:
        blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]
        in_code = False
        code_min_x0 = 0.0
        code_char_width = 6.0

        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue

                text = "".join(s["text"] for s in spans).rstrip()
                if not text.strip():
                    if in_code:
                        output.append("")
                    continue

                all_mono = all(_is_mono(s) for s in spans)
                x0 = line["bbox"][0]
                font_size = spans[0]["size"]

                if all_mono:
                    if not in_code:
                        in_code = True
                        code_min_x0 = x0
                        code_char_width = max(font_size * 0.6, 1.0)
                        output.append("\n```")

                    indent = int(round((x0 - code_min_x0) / code_char_width))
                    output.append(" " * indent + text)
                else:
                    if in_code:
                        in_code = False
                        output.append("```\n")

                    all_bold = all(_is_bold(s) for s in spans)
                    if all_bold and font_size >= 11 and len(text.split()) <= 20:
                        output.append(f"\n## {text}\n")
                    elif all_bold and font_size >= 14:
                        output.append(f"\n# {text}\n")
                    else:
                        output.append(text)

        if in_code:
            in_code = False
            output.append("```\n")

    doc.close()
    return "\n".join(output)


def _extract_with_marker(path: str) -> str | None:
    """OCR fallback for scanned documents."""
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError:
        return None
    try:
        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(path)
        return rendered.markdown
    except Exception:
        return None


def _split_markdown_sections(md: str) -> list[dict]:
    """Split markdown text by headers (# / ## / ###)."""
    matches = list(_HEADER_RE.finditer(md))

    if not matches:
        text = md.strip()
        return [{"title": "(untitled)", "text": text}] if text else []

    sections: list[dict] = []

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


_MIN_CHARS = 100


def extract_pdf(path: str) -> ExtractedDocument:
    """Extract text from a PDF file.

    Strategy:
    1. Font-aware extraction: detects headings via bold/size, code via monospace.
    2. If result is too short (<100 chars), fall back to Marker with OCR.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    md = _extract_pages(path)

    if len(md.strip()) < _MIN_CHARS:
        marker_md = _extract_with_marker(path)
        if marker_md and len(marker_md.strip()) >= _MIN_CHARS:
            md = marker_md

    sections = _split_markdown_sections(md)

    metadata = {
        "filename": p.name,
        "size_bytes": p.stat().st_size,
        "extractor": "pymupdf-fontaware",
    }

    return ExtractedDocument(
        sections=sections,
        metadata=metadata,
        source_type="pdf",
        file_path=path,
    )
