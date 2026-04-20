"""Web page extraction — trafilatura for content extraction."""

from __future__ import annotations

import re

from . import ExtractedDocument


def extract_web(url: str) -> ExtractedDocument:
    """Extract clean text from a web page URL."""
    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return ExtractedDocument(
            sections=[{"title": "", "text": f"ERROR: Failed to fetch {url}"}],
            metadata={"url": url},
            source_type="web",
        )

    result = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        output_format="markdown",
    )

    # Get metadata
    meta_dict: dict = {"url": url}
    try:
        meta = trafilatura.extract(downloaded, output_format="xmltei", with_metadata=True)
        if meta:
            title_m = re.search(r"<title[^>]*>([^<]+)</title>", meta)
            if title_m:
                meta_dict["page_title"] = title_m.group(1)
            author_m = re.search(r"<author>([^<]+)</author>", meta)
            if author_m:
                meta_dict["author"] = author_m.group(1)
    except Exception:
        pass

    if not result:
        return ExtractedDocument(
            sections=[{"title": "", "text": f"ERROR: No content extracted from {url}"}],
            metadata=meta_dict,
            source_type="web",
        )

    sections = _split_web_content(result, meta_dict.get("page_title", ""))

    return ExtractedDocument(
        sections=sections,
        metadata=meta_dict,
        source_type="web",
        file_path=url,
    )


def _split_web_content(text: str, page_title: str) -> list[dict]:
    """Split extracted web content into sections."""
    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    headers = list(header_pattern.finditer(text))

    if headers:
        sections: list[dict] = []
        pre_header = text[: headers[0].start()].strip()
        if pre_header:
            sections.append({"title": page_title, "text": pre_header})

        for i, match in enumerate(headers):
            title = match.group(2).strip()
            start = match.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append({"title": title, "text": section_text})
        return sections

    # No headers — group paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [{"title": page_title, "text": text.strip()}]

    sections = []
    current_parts: list[str] = []
    current_wc = 0

    for para in paragraphs:
        wc = len(para.split())
        if current_wc + wc > 400 and current_parts:
            sections.append({"title": "", "text": "\n\n".join(current_parts)})
            current_parts = []
            current_wc = 0
        current_parts.append(para)
        current_wc += wc

    if current_parts:
        sections.append({"title": "", "text": "\n\n".join(current_parts)})

    if sections and not sections[0]["title"]:
        sections[0]["title"] = page_title

    return sections
