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
    for item_id, _ in book.spine:
        item = book.get_item_with_id(item_id)
        if not item:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        body = soup.find("body") or soup
        sub_sections = _split_by_headings(body)
        if sub_sections:
            sections.extend(sub_sections)
        else:
            text = body.get_text(separator="\n", strip=True)
            if text.strip():
                heading = soup.find(re.compile(r"^h[1-3]$"))
                title = heading.get_text(strip=True) if heading else ""
                sections.append({"title": title, "text": text})

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
    """Split HTML content into sections by h1-h3 tags (recursive search)."""
    from bs4 import NavigableString, Tag

    headings = soup.find_all(re.compile(r"^h[1-3]$"))
    if not headings:
        return []

    sections: list[dict] = []
    for h in headings:
        title = h.get_text(strip=True)
        parts: list[str] = []
        sibling = h.next_sibling
        while sibling:
            if isinstance(sibling, Tag):
                if re.match(r"^h[1-4]$", sibling.name or ""):
                    break
                if sibling.name == "pre":
                    parts.append(f"```\n{sibling.get_text()}\n```")
                else:
                    if sibling.find(re.compile(r"^h[1-3]$")):
                        break
                    text = sibling.get_text(separator="\n", strip=True)
                    if text:
                        parts.append(text)
            elif isinstance(sibling, NavigableString):
                text = str(sibling).strip()
                if text:
                    parts.append(text)
            sibling = sibling.next_sibling

        body_text = "\n\n".join(parts)
        if body_text.strip():
            sections.append({"title": title, "text": body_text})

    return sections
