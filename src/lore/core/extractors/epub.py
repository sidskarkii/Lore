"""EPUB extraction — chapter-aware text extraction using ebooklib."""

from __future__ import annotations

import re
from pathlib import Path

from . import ExtractedDocument


def extract_epub(path: str) -> ExtractedDocument:
    """Extract text from EPUB, splitting by chapters and sub-headings."""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"EPUB not found: {path}")

    book = epub.read_epub(str(p.resolve()), options={"ignore_ncx": True})

    sections: list[dict] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")

        # Chapter title from first heading
        heading = soup.find(re.compile(r"^h[1-3]$"))
        title = heading.get_text(strip=True) if heading else ""

        body = soup.find("body") or soup

        # Split by sub-headings within the chapter
        sub_sections = _split_by_headings(body)
        if sub_sections:
            if not sub_sections[0]["title"] and title:
                sub_sections[0]["title"] = title
            sections.extend(sub_sections)
        else:
            text = body.get_text(separator="\n", strip=True)
            if text.strip():
                sections.append({"title": title, "text": text})

    # Filter out very short sections (navigation, copyright, etc.)
    sections = [s for s in sections if len(s["text"].split()) > 10]

    metadata: dict = {"filename": p.name}
    dc_title = book.get_metadata("DC", "title")
    if dc_title:
        metadata["book_title"] = dc_title[0][0]
    dc_creator = book.get_metadata("DC", "creator")
    if dc_creator:
        metadata["author"] = dc_creator[0][0]
    dc_language = book.get_metadata("DC", "language")
    if dc_language:
        metadata["language"] = dc_language[0][0]

    return ExtractedDocument(
        sections=sections,
        metadata=metadata,
        source_type="epub",
        file_path=str(p),
    )


def _split_by_headings(soup) -> list[dict]:
    """Split an HTML body into sections by h1-h3 tags."""
    from bs4 import NavigableString, Tag

    sections: list[dict] = []
    current_title = ""
    current_parts: list[str] = []

    for child in soup.children:
        if isinstance(child, Tag) and re.match(r"^h[1-3]$", child.name):
            if current_parts:
                text = "\n".join(current_parts).strip()
                if text:
                    sections.append({"title": current_title, "text": text})
            current_title = child.get_text(strip=True)
            current_parts = []
        elif isinstance(child, Tag):
            text = child.get_text(separator="\n", strip=True)
            if text:
                current_parts.append(text)
        elif isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                current_parts.append(text)

    if current_parts:
        text = "\n".join(current_parts).strip()
        if text:
            sections.append({"title": current_title, "text": text})

    return sections
